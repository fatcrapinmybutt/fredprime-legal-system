import datetime
import os
import textwrap
import zipfile

root = "/mnt/data/LitigationOS_Diamond_9999+++_Core_D"
os.makedirs(root, exist_ok=True)

# ---------------------------
# litigation_os_advanced_engines.py
# ---------------------------
advanced_engines_code = r'''#!/usr/bin/env python3
"""
litigation_os_advanced_engines.py

Core engine layer for Litigation OS • Diamond 9999+++.

This module is intentionally self-contained and conservative:
- Uses SQLite as the central "brain".
- Ingests JSON manifests from a filesystem tree into event / exhibit tables.
- Computes simple but useful "actionability" scores for events.
- Optionally runs topic modeling and timeline analysis if the required libs exist.
- Can generate Plotly HTML dashboards if Plotly is installed.

It is designed to tolerate partial or unknown JSON structures:
- If it sees clear "event" or "exhibit" hints, it routes accordingly.
- Otherwise it falls back to treating records as generic events with a best-effort description.
"""

from __future__ import annotations

import glob
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# Optional imports: engines fall back gracefully if missing.
try:
    import plotly.express as px  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

try:
    from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
    from sklearn.decomposition import LatentDirichletAllocation  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# ============================================================================
# Config
# ============================================================================

@dataclass
class Config:
    data_root: str
    db_path: str
    dashboard_dir: str
    text_glob: str = "**/*.json"
    min_topic_docs: int = 10
    n_topics: int = 12


# ============================================================================
# ParallelEngine (placeholder; sequential by default)
# ============================================================================

class ParallelEngine:
    """
    Thin abstraction for parallelism. Currently uses a simple sequential map.
    You can extend this to Ray / multiprocessing later without changing callers.
    """

    def __init__(self) -> None:
        # Hook for Ray / multiprocessing if you ever want it.
        self.mode = "sequential"

    def map(self, fn, iterable: Iterable[Any]) -> List[Any]:
        return [fn(x) for x in iterable]


# ============================================================================
# SQLite brain
# ============================================================================

class LitigationDB:
    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT,
                event_type TEXT,
                event_date TEXT,
                description TEXT,
                source_file TEXT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS exhibits (
                id INTEGER PRIMARY KEY AUTAUTOINCREMENT,
                case_id TEXT,
                label TEXT,
                kind TEXT,
                event_date TEXT,
                source_file TEXT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_kind TEXT,
                object_id TEXT,
                score_type TEXT,
                score_value REAL
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_kind TEXT,
                object_id TEXT,
                topic_index INTEGER,
                weight REAL
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS time_series (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT,
                metric_name TEXT,
                ts_date TEXT,
                metric_value REAL
            );
            """
        )

        self.conn.commit()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear_all_data(self) -> None:
        cur = self.conn.cursor()
        for table in ("events", "exhibits", "scores", "topics", "time_series"):
            cur.execute(f"DELETE FROM {table};")
        self.conn.commit()

    # ------------------------------------------------------------------
    # Inserts
    # ------------------------------------------------------------------

    def insert_event(
        self,
        case_id: str,
        event_type: str,
        event_date: str,
        description: str,
        source_file: str,
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO events (case_id, event_type, event_date, description, source_file)
            VALUES (?, ?, ?, ?, ?);
            """,
            (case_id, event_type, event_date, description, source_file),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def insert_exhibit(
        self,
        case_id: str,
        label: str,
        kind: str,
        event_date: str,
        source_file: str,
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO exhibits (case_id, label, kind, event_date, source_file)
            VALUES (?, ?, ?, ?, ?);
            """,
            (case_id, label, kind, event_date, source_file),
        )
        self.conn.commit()
        return int(cur.lastrowid)


# ============================================================================
# Ingestion
# ============================================================================

KEYWORDS_ACTION = [
    "ppo",
    "ex parte",
    "custody",
    "parenting time",
    "parenting-time",
    "eviction",
    "notice",
    "hearing",
    "order",
    "motion",
    "show cause",
    "sanction",
    "contempt",
    "appeal",
    "stay",
]

def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (str, bytes)):
        return str(v)
    return str(v)


def _infer_case_id(rec: Dict[str, Any], default: str = "") -> str:
    for key in ("case_id", "caseId", "case", "docket", "docket_number"):
        if key in rec and rec[key]:
            return _safe_str(rec[key])
    return default


def _infer_date(rec: Dict[str, Any]) -> str:
    for key in ("event_date", "date", "date_str", "ts_date"):
        if key in rec and rec[key]:
            return _safe_str(rec[key])
    return ""


def _infer_event_type(rec: Dict[str, Any]) -> str:
    for key in ("event_type", "type", "kind", "category"):
        if key in rec and rec[key]:
            return _safe_str(rec[key])
    return "event"


def _infer_exhibit_label(rec: Dict[str, Any]) -> str:
    for key in ("label", "name", "title", "exhibit_label"):
        if key in rec and rec[key]:
            return _safe_str(rec[key])
    return ""


def _infer_exhibit_kind(rec: Dict[str, Any]) -> str:
    for key in ("kind", "type", "category"):
        if key in rec and rec[key]:
            return _safe_str(rec[key])
    return "exhibit"


def _infer_description(rec: Dict[str, Any]) -> str:
    for key in ("description", "text", "summary", "body"):
        if key in rec and isinstance(rec[key], str):
            return rec[key]
    # Fallback: compact JSON
    try:
        return json.dumps(rec, ensure_ascii=False)
    except Exception:
        return str(rec)


def _route_record(rec: Dict[str, Any]) -> str:
    """
    Decide whether a record is an 'event' or an 'exhibit' using simple heuristics.
    Defaults to 'event'.
    """
    kind = _safe_str(rec.get("kind") or rec.get("type") or "").lower()
    label = _safe_str(rec.get("label") or "")
    if "exhibit" in kind:
        return "exhibit"
    if label.lower().startswith("exhibit"):
        return "exhibit"
    if "hearing" in kind or "motion" in kind or "order" in kind or "event" in kind:
        return "event"
    return "event"


def _flatten_records(obj: Any) -> List[Dict[str, Any]]:
    """
    Try to extract a list of dict-like records from arbitrary JSON.
    """
    records: List[Dict[str, Any]] = []

    if isinstance(obj, dict):
        # Common pattern: {"events": [...], "exhibits": [...]}
        for key in ("events", "exhibits", "nodes", "items", "records"):
            if key in obj and isinstance(obj[key], list):
                for item in obj[key]:
                    if isinstance(item, dict):
                        records.append(item)
        # If we didn't find anything, treat the dict as a single record.
        if not records:
            records.append(obj)

    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                records.append(item)

    return records


def ingest_json_nodes_into_db(cfg: Config, db: LitigationDB, parallel: ParallelEngine) -> None:
    """
    Scan cfg.data_root recursively for JSON files and ingest them into the DB.
    """
    pattern = os.path.join(cfg.data_root, cfg.text_glob)
    paths = glob.glob(pattern, recursive=True)

    print(f"[INGEST] Data root: {cfg.data_root}")
    print(f"[INGEST] Pattern:   {pattern}")
    print(f"[INGEST] JSON files discovered: {len(paths)}")

    db.clear_all_data()

    def process_path(path: str) -> Tuple[int, int]:
        events = 0
        exhibits = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[INGEST] Failed to parse JSON: {path} ({exc})")
            return 0, 0

        records = _flatten_records(data)
        if not records:
            return 0, 0

        source_file = os.path.abspath(path)
        for rec in records:
            route = _route_record(rec)
            case_id = _infer_case_id(rec)
            event_date = _infer_date(rec)

            if route == "event":
                event_type = _infer_event_type(rec)
                description = _infer_description(rec)
                db.insert_event(case_id, event_type, event_date, description, source_file)
                events += 1
            else:
                label = _infer_exhibit_label(rec)
                kind = _infer_exhibit_kind(rec)
                db.insert_exhibit(case_id, label or kind, kind, event_date, source_file)
                exhibits += 1

        return events, exhibits

    total_events = 0
    total_exhibits = 0

    for ev, ex in parallel.map(process_path, paths):
        total_events += ev
        total_exhibits += ex

    print(f"[INGEST] Inserted events:   {total_events}")
    print(f"[INGEST] Inserted exhibits: {total_exhibits}")


# ============================================================================
# Actionability ranker
# ============================================================================

class ActionabilityRanker:
    """
    Simple heuristic actionability scorer.

    Score inputs:
    - Length of description.
    - Presence of key litigation terms (PPO, ex parte, custody, eviction, etc.).
    Output:
    - scores table rows with score_type="actionability" and score_value in [0, 1].
    """

    def score_events(self, db: LitigationDB) -> None:
        cur = db.conn.cursor()

        cur.execute("DELETE FROM scores WHERE score_type = 'actionability';")
        db.conn.commit()

        events_df = pd.read_sql_query(
            "SELECT id, case_id, event_type, event_date, description FROM events;",
            db.conn,
        )

        if events_df.empty:
            print("[RANK] No events found; skipping actionability scoring.")
            return

        scores: List[Tuple[str, str, float]] = []

        for _, row in events_df.iterrows():
            ev_id = str(row["id"])
            desc = row.get("description") or ""
            desc_low = str(desc).lower()
            base = min(len(desc) / 1000.0, 3.0)  # 0..3
            hits = 0
            for kw in KEYWORDS_ACTION:
                if kw in desc_low:
                    hits += 1
            keyword_boost = min(hits * 0.5, 5.0)
            raw = base + keyword_boost
            scores.append(("event", ev_id, float(raw)))

        raw_values = np.array([s[2] for s in scores], dtype="float32")
        if raw_values.size == 0:
            print("[RANK] No scores to normalize.")
            return

        min_val = float(raw_values.min())
        max_val = float(raw_values.max())
        span = max_val - min_val if max_val > min_val else 1.0

        normalized: List[Tuple[str, str, float]] = []
        for kind, obj_id, raw in scores:
            norm = (raw - min_val) / span
            normalized.append((kind, obj_id, float(norm)))

        cur.executemany(
            """
            INSERT INTO scores (object_kind, object_id, score_type, score_value)
            VALUES (?, ?, 'actionability', ?);
            """,
            [(k, i, v) for (k, i, v) in normalized],
        )
        db.conn.commit()

        print(f"[RANK] Scored {len(normalized)} events with actionability in [0, 1].")


# ============================================================================
# Topic modeling engine
# ============================================================================

class TopicModelEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self, db: LitigationDB) -> None:
        if not _HAS_SKLEARN:
            print("[TOPIC] scikit-learn not available; skipping topic modeling.")
            return

        events_df = pd.read_sql_query(
            "SELECT id, description FROM events;",
            db.conn,
        )
        if len(events_df) < self.cfg.min_topic_docs:
            print(f"[TOPIC] Only {len(events_df)} events; need at least {self.cfg.min_topic_docs}. Skipping.")
            return

        texts = [
            str(desc) if isinstance(desc, str) else ""
            for desc in events_df["description"].tolist()
        ]

        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=5000,
            stop_words="english",
        )
        X = vectorizer.fit_transform(texts)

        n_topics = max(2, self.cfg.n_topics)
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method="batch",
            max_iter=20,
            random_state=0,
        )
        topic_dist = lda.fit_transform(X)  # shape: (n_docs, n_topics)

        cur = db.conn.cursor()
        cur.execute("DELETE FROM topics;")

        for (doc_idx, ev_id), dist_row in zip(events_df[["id"]].iterrows(), topic_dist):
            ev_id_str = str(ev_id["id"])
            for topic_index, weight in enumerate(dist_row):
                if weight <= 0:
                    continue
                cur.execute(
                    """
                    INSERT INTO topics (object_kind, object_id, topic_index, weight)
                    VALUES ('event', ?, ?, ?);
                    """,
                    (ev_id_str, int(topic_index), float(weight)),
                )

        db.conn.commit()
        print(f"[TOPIC] Wrote topic weights for {len(events_df)} events across {n_topics} topics.")


# ============================================================================
# Time-series / temporal engine
# ============================================================================

class TemporalSignalEngine:
    """
    Builds basic time-series signals:
    - event_count per (case_id, date)
    """

    def run(self, db: LitigationDB) -> None:
        events_df = pd.read_sql_query(
            "SELECT case_id, event_date FROM events WHERE event_date IS NOT NULL AND event_date != '';",
            db.conn,
        )
        if events_df.empty:
            print("[TIME] No dated events; skipping time-series build.")
            return

        events_df["case_id"] = events_df["case_id"].fillna("").astype(str)
        events_df["event_date"] = events_df["event_date"].fillna("").astype(str)

        grouped = (
            events_df.groupby(["case_id", "event_date"])
            .size()
            .reset_index(name="event_count")
        )

        cur = db.conn.cursor()
        cur.execute("DELETE FROM time_series;")

        rows = []
        for _, row in grouped.iterrows():
            case_id = row["case_id"]
            ts_date = row["event_date"]
            value = float(row["event_count"])
            rows.append((case_id, "event_count", ts_date, value))

        cur.executemany(
            """
            INSERT INTO time_series (case_id, metric_name, ts_date, metric_value)
            VALUES (?, ?, ?, ?);
            """,
            rows,
        )
        db.conn.commit()

        print(f"[TIME] Wrote {len(rows)} time-series points (event_count).")


# ============================================================================
# Dashboards
# ============================================================================

class DashboardEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        os.makedirs(self.cfg.dashboard_dir, exist_ok=True)

    def build_event_timeline(self, db: LitigationDB) -> None:
        if not _HAS_PLOTLY:
            print("[DASH] Plotly not available; skipping event timeline dashboard.")
            return

        df = pd.read_sql_query(
            """
            SELECT e.id, e.case_id, e.event_type, e.event_date, e.source_file,
                   COALESCE(s.score_value, 0.0) AS actionability
            FROM events AS e
            LEFT JOIN scores AS s
              ON s.object_kind = 'event'
             AND s.object_id = CAST(e.id AS TEXT)
             AND s.score_type = 'actionability';
            """,
            db.conn,
        )

        if df.empty:
            print("[DASH] No events; skipping event timeline.")
            return

        df["case_id"] = df["case_id"].fillna("").astype(str)
        df["event_date"] = df["event_date"].fillna("").astype(str)

        out_path = os.path.join(self.cfg.dashboard_dir, "events_timeline.html")

        import plotly.express as px

        fig = px.scatter(
            df,
            x="event_date",
            y="case_id",
            size="actionability",
            color="case_id",
            hover_data=["event_type", "source_file"],
            title="Litigation OS • Event Timeline (Actionability-sized)",
        )
        fig.update_layout(autosize=True, height=800)
        fig.write_html(out_path)
        print(f"[DASH] Wrote event timeline dashboard: {out_path}")

    def build_time_series_heatmap(self, db: LitigationDB) -> None:
        if not _HAS_PLOTLY:
            print("[DASH] Plotly not available; skipping time-series heatmap dashboard.")
            return

        df = pd.read_sql_query(
            """
            SELECT case_id, ts_date, metric_value
            FROM time_series
            WHERE metric_name = 'event_count';
            """,
            db.conn,
        )

        if df.empty:
            print("[DASH] No time-series data; skipping heatmap.")
            return

        df["case_id"] = df["case_id"].fillna("").astype(str)
        df["ts_date"] = df["ts_date"].fillna("").astype(str)

        pivot = df.pivot_table(
            index="case_id",
            columns="ts_date",
            values="metric_value",
            fill_value=0.0,
        )

        out_path = os.path.join(self.cfg.dashboard_dir, "time_series_heatmap.html")

        import plotly.express as px

        fig = px.imshow(
            pivot.values,
            labels=dict(x="Date", y="Case ID", color="Event count"),
            x=list(pivot.columns),
            y=list(pivot.index),
            title="Litigation OS • Event Count Heatmap",
            aspect="auto",
        )
        fig.update_xaxes(side="bottom")
        fig.write_html(out_path)
        print(f"[DASH] Wrote time-series heatmap dashboard: {out_path}")


# ============================================================================
# Convenience: full pipeline from config
# ============================================================================

def run_pipeline(cfg: Config) -> None:
    """
    Convenience wrapper to run all engines without the external orchestrator.
    The orchestrator (litigation_os_orchestrator.py) uses the lower-level pieces
    directly, but this is useful for quick tests.
    """
    print("[ENGINES] Starting Litigation OS core pipeline.")
    parallel = ParallelEngine()
    db = LitigationDB(cfg.db_path)

    ingest_json_nodes_into_db(cfg, db, parallel)

    ranker = ActionabilityRanker()
    ranker.score_events(db)

    topic_engine = TopicModelEngine(cfg)
    topic_engine.run(db)

    time_engine = TemporalSignalEngine()
    time_engine.run(db)

    dash_engine = DashboardEngine(cfg)
    dash_engine.build_event_timeline(db)
    dash_engine.build_time_series_heatmap(db)

    print("[ENGINES] Core pipeline complete.")
'''

# ---------------------------
# litigation_os_orchestrator.py
# ---------------------------
orchestrator_code = r'''#!/usr/bin/env python3
"""
litigation_os_orchestrator.py

Top-level orchestrator for the upgraded Litigation OS stack, with a simple
progress bar across all major stages.

Pipeline:
1. Ingest JSON nodes/exhibits into SQLite via advanced engines.
2. Rank events by actionability (heuristic scoring).
3. Run topic modeling over narrative payloads (if scikit-learn is available).
4. Build time-series signals and basic metrics.
5. Generate HTML dashboards (Plotly) from the DB (if Plotly is available).
6. Build an upgraded graph snapshot (nodes/edges) from the DB.

This file must live in the same directory as:
  - litigation_os_advanced_engines.py
  - litigation_os_launcher.py
"""

import argparse
import os
import json
import sqlite3
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

import litigation_os_advanced_engines as adv


# =====================================================================
# Simple ASCII progress bar
# =====================================================================

class ProgressBar:
    def __init__(self, total_steps: int, bar_width: int = 30) -> None:
        self.total_steps = max(1, total_steps)
        self.bar_width = bar_width
        self.current_step = 0

    def advance(self, label: str = "") -> None:
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps

        frac = self.current_step / self.total_steps
        filled = int(self.bar_width * frac)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        pct = int(frac * 100)

        print(f"\r[PROGRESS] [{bar}] {self.current_step}/{self.total_steps} {pct}% - {label}", end="", flush=True)

    def done(self) -> None:
        print()


# =====================================================================
# Graph builder from SQLite brain
# =====================================================================

def build_graph_from_db(db_path: str, graph_dir: str) -> None:
    """
    Read core tables from the SQLite brain and emit:
      - nodes_advanced.json
      - edges_advanced.json

    Nodes:
      - type="case":    one per case_id
      - type="event":   one per events.id
      - type="exhibit": one per exhibits.id

    Edges:
      - case_has_event:           case -> event
      - case_has_exhibit:         case -> exhibit
      - exhibit_related_to_event: exhibit -> event (same case_id and date)
    """
    os.makedirs(graph_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)

    events_df = pd.read_sql_query("SELECT * FROM events;", conn)
    exhibits_df = pd.read_sql_query("SELECT * FROM exhibits;", conn)
    scores_df = pd.read_sql_query(
        "SELECT object_id, object_kind, score_type, score_value FROM scores;",
        conn,
    )
    topics_df = pd.read_sql_query(
        "SELECT object_id, object_kind, topic_index, weight FROM topics;",
        conn,
    )

    if events_df.empty and exhibits_df.empty:
        print("[GRAPH] No events or exhibits in DB; nothing to graph.")
        return

    # -----------------------------------------------------------------
    # Build lookups for actionability scores and topics
    # -----------------------------------------------------------------
    action_scores: Dict[Tuple[str, str], float] = {}
    if not scores_df.empty:
        scores_action = scores_df[scores_df["score_type"] == "actionability"]
        for _, row in scores_action.iterrows():
            key = (str(row["object_kind"]), str(row["object_id"]))
            action_scores[key] = float(row["score_value"])

    topics_map: Dict[Tuple[str, str], List[Tuple[int, float]]] = defaultdict(list)
    if not topics_df.empty:
        for _, row in topics_df.iterrows():
            key = (str(row["object_kind"]), str(row["object_id"]))
            topics_map[key].append((int(row["topic_index"]), float(row["weight"])))

        for key, lst in topics_map.items():
            lst.sort(key=lambda x: x[1], reverse=True)

    nodes: List[Dict] = []
    edges: List[Dict] = []

    # -----------------------------------------------------------------
    # Case nodes
    # -----------------------------------------------------------------
    case_ids = set()
    if not events_df.empty:
        case_ids.update([c for c in events_df["case_id"].astype(str).tolist() if c])
    if not exhibits_df.empty:
        case_ids.update([c for c in exhibits_df["case_id"].astype(str).tolist() if c])

    for cid in sorted(case_ids):
        node_id = f"case:{cid}"
        nodes.append({
            "id": node_id,
            "label": cid,
            "type": "case",
            "case_id": cid,
            "actionability_score": None,
            "top_topics": [],
        })

    # -----------------------------------------------------------------
    # Helper: topic list for an object
    # -----------------------------------------------------------------
    def get_top_topics(kind: str, obj_id: str, top_n: int = 3) -> List[Dict]:
        lst = topics_map.get((kind, obj_id), [])
        lst = lst[:top_n]
        return [{"topic_index": t, "weight": float(w)} for t, w in lst]

    # -----------------------------------------------------------------
    # Event nodes + edges (case -> event)
    # -----------------------------------------------------------------
    if not events_df.empty:
        for _, row in events_df.iterrows():
            ev_id = str(row["id"])
            case_id = str(row.get("case_id") or "")
            event_type = str(row.get("event_type") or "event")
            event_date = str(row.get("event_date") or "")
            source_file = str(row.get("source_file") or "")

            node_id = f"event:{ev_id}"
            score = action_scores.get(("event", ev_id))
            top_topics = get_top_topics("event", ev_id)

            nodes.append({
                "id": node_id,
                "label": f"{event_type} {event_date}".strip(),
                "type": "event",
                "case_id": case_id,
                "event_type": event_type,
                "event_date": event_date,
                "source_file": source_file,
                "actionability_score": float(score) if score is not None else None,
                "top_topics": top_topics,
            })

            if case_id:
                case_node_id = f"case:{case_id}"
                edges.append({
                    "id": f"edge:case:{case_id}:event:{ev_id}",
                    "source": case_node_id,
                    "target": node_id,
                    "type": "case_has_event",
                })

    # -----------------------------------------------------------------
    # Exhibit nodes + edges (case -> exhibit)
    # -----------------------------------------------------------------
    if not exhibits_df.empty:
        for _, row in exhibits_df.iterrows():
            ex_id = str(row["id"])
            case_id = str(row.get("case_id") or "")
            label = str(row.get("label") or "")
            kind = str(row.get("kind") or "exhibit")
            event_date = str(row.get("event_date") or "")
            source_file = str(row.get("source_file") or "")

            node_id = f"exhibit:{ex_id}"
            score = action_scores.get(("exhibit", ex_id))
            top_topics = get_top_topics("exhibit", ex_id)

            nodes.append({
                "id": node_id,
                "label": label if label else f"{kind} {ex_id}",
                "type": "exhibit",
                "kind": kind,
                "case_id": case_id,
                "event_date": event_date,
                "source_file": source_file,
                "actionability_score": float(score) if score is not None else None,
                "top_topics": top_topics,
            })

            if case_id:
                case_node_id = f"case:{case_id}"
                edges.append({
                    "id": f"edge:case:{case_id}:exhibit:{ex_id}",
                    "source": case_node_id,
                    "target": node_id,
                    "type": "case_has_exhibit",
                })

    # -----------------------------------------------------------------
    # Exhibit -> Event edges (same case_id and event_date)
    # -----------------------------------------------------------------
    if not exhibits_df.empty and not events_df.empty:
        events_df["case_id"] = events_df["case_id"].astype(str)
        events_df["event_date"] = events_df["event_date"].astype(str)
        idx_events: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for _, row in events_df.iterrows():
            cid = str(row.get("case_id") or "")
            d = str(row.get("event_date") or "")
            if cid and d:
                idx_events[(cid, d)].append(str(row["id"]))

        for _, row in exhibits_df.iterrows():
            ex_id = str(row["id"])
            cid = str(row.get("case_id") or "")
            d = str(row.get("event_date") or "")
            if not cid or not d:
                continue
            key = (cid, d)
            if key not in idx_events:
                continue
            for ev_id in idx_events[key]:
                edges.append({
                    "id": f"edge:exhibit:{ex_id}:event:{ev_id}",
                    "source": f"exhibit:{ex_id}",
                    "target": f"event:{ev_id}",
                    "type": "exhibit_related_to_event",
                })

    # -----------------------------------------------------------------
    # Write graph outputs
    # -----------------------------------------------------------------
    nodes_path = os.path.join(graph_dir, "nodes_advanced.json")
    edges_path = os.path.join(graph_dir, "edges_advanced.json")

    with open(nodes_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open(edges_path, "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    print(f"\n[GRAPH] Wrote {len(nodes)} nodes to {nodes_path}")
    print(f"[GRAPH] Wrote {len(edges)} edges to {edges_path}")


# =====================================================================
# Orchestration (engines + graph) with progress bar
# =====================================================================

def run_full_pipeline(
    data_root: str,
    db_path: str,
    dashboard_dir: str,
    graph_dir: str,
    text_glob: str = "**/*.json",
    min_topic_docs: int = 10,
    n_topics: int = 12,
) -> None:
    """
    Full Litigation OS pipeline:

      1. Ingest JSON nodes/exhibits into DB.
      2. Rank events by actionability.
      3. Topic modeling.
      4. Time-series / metrics.
      5. Dashboards.
      6. Graph snapshot from DB.

    Each major step updates a simple ASCII progress bar.
    """
    print("[ORCH] Starting full Litigation OS pipeline.")
    print(f"[ORCH] data_root    = {data_root}")
    print(f"[ORCH] db_path      = {db_path}")
    print(f"[ORCH] dashboardDir = {dashboard_dir}")
    print(f"[ORCH] graphDir     = {graph_dir}")

    cfg = adv.Config(
        data_root=data_root,
        db_path=db_path,
        dashboard_dir=dashboard_dir,
        text_glob=text_glob,
        min_topic_docs=min_topic_docs,
        n_topics=n_topics,
    )

    # Initialize engines
    parallel = adv.ParallelEngine()
    db = adv.LitigationDB(cfg.db_path)
    ranker = adv.ActionabilityRanker()
    topic_engine = adv.TopicModelEngine(cfg)
    time_engine = adv.TemporalSignalEngine()
    dash_engine = adv.DashboardEngine(cfg)

    # 6 major steps
    pb = ProgressBar(total_steps=6)

    # 1. Ingest JSON into DB
    pb.advance("Ingesting JSON nodes/exhibits into DB")
    adv.ingest_json_nodes_into_db(cfg, db, parallel)

    # 2. Rank events by actionability
    pb.advance("Ranking events by actionability")
    ranker.score_events(db)

    # 3. Topic modeling
    pb.advance("Running topic modeling")
    topic_engine.run(db)

    # 4. Time-series / metrics
    pb.advance("Building time-series signals")
    time_engine.run(db)

    # 5. Dashboards
    pb.advance("Generating dashboards")
    dash_engine.build_event_timeline(db)
    dash_engine.build_time_series_heatmap(db)

    # 6. Graph snapshot
    pb.advance("Building graph from DB")
    build_graph_from_db(cfg.db_path, graph_dir)

    pb.done()
    print("[ORCH] Full pipeline complete.")


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Litigation OS Orchestrator with progress bar")
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root containing JSON node/exhibit manifests "
             "(e.g. D:/LitigationOS/NODES or D:/)",
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="SQLite DB path, e.g. D:/LitigationOS/litigation_os.db",
    )
    parser.add_argument(
        "--dashboard-dir",
        required=True,
        help="Folder for Plotly HTML dashboards (e.g. D:/LitigationOS/DASHBOARDS)",
    )
    parser.add_argument(
        "--graph-dir",
        required=True,
        help="Folder for graph JSON outputs (e.g. D:/LitigationOS/GRAPHS/ADVANCED)",
    )
    parser.add_argument(
        "--text-glob",
        default="**/*.json",
        help="Glob pattern for JSON files under data-root (default: **/*.json)",
    )
    parser.add_argument(
        "--min-topic-docs",
        type=int,
        default=10,
        help="Minimum documents required to run topic modeling (default: 10)",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=12,
        help="Number of topics for topic modeling (default: 12)",
    )
    return parser


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    run_full_pipeline(
        data_root=args.data_root,
        db_path=args.db_path,
        dashboard_dir=args.dashboard_dir,
        graph_dir=args.graph_dir,
        text_glob=args.text_glob,
        min_topic_docs=args.min_topic_docs,
        n_topics=args.n_topics,
    )
'''

# ---------------------------
# litigation_os_launcher.py (D:\LitigationOS defaults)
# ---------------------------
launcher_code = r'''#!/usr/bin/env python3
"""
litigation_os_launcher.py

One-command launcher for the upgraded Litigation OS pipeline.

Stack:
- litigation_os_advanced_engines.py
- litigation_os_orchestrator.py

On Windows, defaults now assume:
    D:/LitigationOS/NODES
    D:/LitigationOS/litigation_os.db
    D:/LitigationOS/DASHBOARDS
    D:/LitigationOS/GRAPHS/ADVANCED

On Android/Termux, defaults assume:
    /storage/emulated/0/Download/LitigationOS/NODES
    /storage/emulated/0/Download/LitigationOS/litigation_os.db
    /storage/emulated/0/Download/LitigationOS/DASHBOARDS
    /storage/emulated/0/Download/LitigationOS/GRAPHS/ADVANCED

You can override everything via CLI flags.
"""

import argparse
import os
import sys
from typing import Tuple

from litigation_os_orchestrator import run_full_pipeline


def detect_defaults() -> Tuple[str, str, str, str]:
    """
    Detect environment and return:
      data_root, db_path, dashboard_dir, graph_dir
    """
    is_windows = os.name == "nt"

    if is_windows:
        # Canonical Litigation OS root on your system
        base = "D:/LitigationOS"
    else:
        base = "/storage/emulated/0/Download/LitigationOS"

    data_root = os.path.join(base, "NODES")
    db_path = os.path.join(base, "litigation_os.db")
    dashboard_dir = os.path.join(base, "DASHBOARDS")
    graph_dir = os.path.join(base, "GRAPHS", "ADVANCED")

    return data_root, db_path, dashboard_dir, graph_dir


def parse_args() -> argparse.Namespace:
    d_data_root, d_db_path, d_dash_dir, d_graph_dir = detect_defaults()

    parser = argparse.ArgumentParser(
        description="One-command launcher for Litigation OS full pipeline"
    )
    parser.add_argument(
        "--data-root",
        default=d_data_root,
        help=f"Root containing JSON nodes/exhibits (default: {d_data_root})",
    )
    parser.add_argument(
        "--db-path",
        default=d_db_path,
        help=f"SQLite DB path (default: {d_db_path})",
    )
    parser.add_argument(
        "--dashboard-dir",
        default=d_dash_dir,
        help=f"Plotly dashboards dir (default: {d_dash_dir})",
    )
    parser.add_argument(
        "--graph-dir",
        default=d_graph_dir,
        help=f"Graph JSON output dir (default: {d_graph_dir})",
    )
    parser.add_argument(
        "--text-glob",
        default="**/*.json",
        help="Glob pattern for JSON files (default: **/*.json)",
    )
    parser.add_argument(
        "--min-topic-docs",
        type=int,
        default=10,
        help="Minimum docs for topic modeling (default: 10)",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=12,
        help="Number of topics for topic modeling (default: 12)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[LAUNCH] Litigation OS full pipeline")
    print(f"[LAUNCH] data_root    = {args.data_root}")
    print(f"[LAUNCH] db_path      = {args.db_path}")
    print(f"[LAUNCH] dashboardDir = {args.dashboard_dir}")
    print(f"[LAUNCH] graphDir     = {args.graph_dir}")
    sys.stdout.flush()

    run_full_pipeline(
        data_root=args.data_root,
        db_path=args.db_path,
        dashboard_dir=args.dashboard_dir,
        graph_dir=args.graph_dir,
        text_glob=args.text_glob,
        min_topic_docs=args.min_topic_docs,
        n_topics=args.n_topics,
    )

    print("[LAUNCH] Pipeline finished.")


if __name__ == "__main__":
    main()
'''

# ---------------------------
# litigation_os_chat_extractor_local.py
# ---------------------------
chat_extractor_code = r'''#!/usr/bin/env python3
"""
litigation_os_chat_extractor_local.py

Purpose:
    Process a ZIP of chat logs (JSON/TXT/HTML) and extract:
      1. All code/script blocks across all conversations.
      2. All user narrative (your side only), for affidavits and context.
      3. Auto-flagged "important" snippets for a knowledge core.

Hard-wired settings for Andrew J. Pigors:

    ZIP_PATH  = "C:\\Users\\andre\\Downloads\\5a196c922797e6ddc522eff6e8a17f0702237f861aab8c32f570470398c9aa12-2025-11-04-15-30-02-f42b3afb66d84e5396c1a4630be069a4.zip"
    OUT_DIR   = "C:\\Users\\andre\\Downloads\\chat_extract_out"
    MODE      = "all"   # options: "all", "code", "narrative", "knowledge"

How to use (Windows):

    1. Ensure Python 3 is installed and in PATH.
    2. Save this file anywhere (for this bundle, it sits next to the other Litigation OS scripts).
    3. Open Command Prompt and run from that folder:
           python litigation_os_chat_extractor_local.py
    4. Outputs will appear under:
           C:\\Users\\andre\\Downloads\\chat_extract_out

Outputs created (depending on MODE):

    code_blocks_all.txt
    code_blocks_all.json
    user_narrative_all.txt
    user_narrative_all.json
    knowledge_snippets_all.txt
    knowledge_snippets_all.json
"""

import json
import os
import re
import zipfile
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# HARD-WIRED CONFIG FOR YOUR SYSTEM
# ---------------------------------------------------------------------------

ZIP_PATH = r"C:\Users\andre\Downloads\5a196c922797e6ddc522eff6e8a17f0702237f861aab8c32f570470398c9aa12-2025-11-04-15-30-02-f42b3afb66d84e5396c1a4630be069a4.zip"
OUT_DIR = r"C:\Users\andre\Downloads\chat_extract_out"

# MODE options:
#   "all"        = run all three layers (code + narrative + knowledge)
#   "code"       = only code/script layer
#   "narrative"  = only user narrative layer
#   "knowledge"  = only important-snippet layer (requires narrative pass)
MODE = "all"

# ---------------------------------------------------------------------------
# PROGRESS BAR
# ---------------------------------------------------------------------------

class ProgressBar:
    def __init__(self, total_steps: int, bar_width: int = 30) -> None:
        self.total_steps = max(1, total_steps)
        self.bar_width = bar_width
        self.current_step = 0

    def advance(self, label: str = "") -> None:
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps

        frac = self.current_step / self.total_steps
        filled = int(self.bar_width * frac)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        pct = int(frac * 100)

        print(f"\r[PROGRESS] [{bar}] {self.current_step}/{self.total_steps} {pct}% - {label}", end="", flush=True)

    def done(self) -> None:
        print()


# ---------------------------------------------------------------------------
# INTERNAL CONSTANTS AND REGEXES
# ---------------------------------------------------------------------------

# Regex to capture fenced code blocks: ```lang\n ... ```
CODE_BLOCK_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+\-]*)\s*\n(?P<code>.*?)(```)",
    re.DOTALL,
)

# Heuristic keywords for "important" snippets
IMPORTANT_KEYWORDS = [
    "MCR", "MCL", "Benchbook", "SCAO",
    "Shady Oaks", "Homes of America", "Alden Global",
    "EGLE", "sewage", "utility", "rent", "ledger",
    "PPO", "show cause", "contempt",
    "custody", "parenting time", "14th Circuit", "60th District",
    "friend of the court", "FOC",
    "Emily Watson", "Lori Watson", "Albert Watson", "Lincoln",
    "HealthWest", "mental health", "evaluation",
    "due process", "bias", "judicial", "canon",
    "JTC", "Judicial Tenure Commission",
    "appeal", "COA", "Court of Appeals", "MSC", "Supreme Court", "WDMI",
    "FRED", "Litigation OS", "MEEK", "MindEye2", "graph",
    "rclone", "Termux", "EDS-USB",
]

# Minimum character length for a narrative chunk to be considered substantive
MIN_IMPORTANT_LEN = 400


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def ensure_out_dir(path: str) -> str:
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path


def safe_read_text_from_zip(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> Optional[str]:
    try:
        with zf.open(info, "r") as f:
            data = f.read()
        return data.decode("utf-8", errors="ignore")
    except Exception as exc:
        print(f"\n[WARN] Failed to read {info.filename}: {exc}")
        return None


def is_text_like(filename: str) -> bool:
    filename = filename.lower()
    text_exts = (".json", ".txt", ".md", ".html", ".htm", ".log")
    return filename.endswith(text_exts)


def normalize_whitespace(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def extract_code_blocks_from_text(text: str, source: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for m in CODE_BLOCK_RE.finditer(text):
        lang = m.group("lang") or ""
        code = m.group("code").strip("\n\r ")
        if not code:
            continue
        blocks.append(
            {
                "source": source,
                "language": lang,
                "code": code,
            }
        )
    return blocks


def json_walk_collect_messages(
    obj: Any,
    source: str,
    user_messages: List[Dict[str, Any]],
    other_messages: List[Dict[str, Any]],
) -> None:
    """
    Recursive walk for JSON-based exports.
    Heuristics:
        - If dict has author/role + content, treat as a message.
        - user-side: role == "user" or author.role == "user".
    """
    if isinstance(obj, dict):
        author = obj.get("author")
        role = obj.get("role")
        content = obj.get("content")

        resolved_role = None
        if isinstance(author, dict) and "role" in author:
            resolved_role = str(author.get("role"))
        elif isinstance(author, str):
            resolved_role = author
        elif isinstance(role, str):
            resolved_role = role

        text_content = None
        if isinstance(content, dict):
            if content.get("parts") and isinstance(content["parts"], list):
                text_content = "\n".join(
                    str(p) for p in content["parts"] if isinstance(p, str)
                )
            elif "text" in content and isinstance(content["text"], str):
                text_content = content["text"]
        elif isinstance(content, str):
            text_content = content

        if text_content:
            msg = {
                "source": source,
                "role": resolved_role or "unknown",
                "text": normalize_whitespace(text_content).strip(),
            }
            if msg["text"]:
                if (resolved_role or "").lower() == "user":
                    user_messages.append(msg)
                else:
                    other_messages.append(msg)

        for v in obj.values():
            json_walk_collect_messages(v, source, user_messages, other_messages)

    elif isinstance(obj, list):
        for v in obj:
            json_walk_collect_messages(v, source, user_messages, other_messages)


def text_transcript_split_roles(
    text: str,
    source: str,
    user_messages: List[Dict[str, Any]],
    other_messages: List[Dict[str, Any]],
) -> None:
    """
    Heuristic role splitter for plain-text transcripts.

    Looks for lines starting with markers:
        "User:", "Andrew:", "Assistant:", "GPT:", etc.
    Groups subsequent lines until the next marker.
    """
    lines = normalize_whitespace(text).split("\n")
    current_role: Optional[str] = None
    buffer: List[str] = []

    def flush():
        nonlocal buffer, current_role
        if not buffer:
            return
        msg_text = "\n".join(buffer).strip()
        if not msg_text:
            buffer = []
            return
        msg = {"source": source, "role": current_role or "unknown", "text": msg_text}
        if current_role == "user":
            user_messages.append(msg)
        elif current_role is None:
            # If role is unknown, bias toward user narrative for affidavit harvesting
            user_messages.append(msg)
        else:
            other_messages.append(msg)
        buffer = []

    user_markers = ("user:", "andrew:", "andrew j pigors:", "andrew j. pigors:")
    assistant_markers = ("assistant:", "gpt:", "chatgpt:", "model:")

    for raw_line in lines:
        line = raw_line.strip()
        low = line.lower()
        new_role: Optional[str] = None

        for m in user_markers:
            if low.startswith(m):
                new_role = "user"
                line = line[len(m):].lstrip()
                break

        if new_role is None:
            for m in assistant_markers:
                if low.startswith(m):
                    new_role = "assistant"
                    line = line[len(m):].lstrip()
                    break

        if new_role is not None:
            flush()
            current_role = new_role
            if line:
                buffer.append(line)
        else:
            buffer.append(raw_line)

    flush()


def build_important_snippets(
    user_messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    snippets: List[Dict[str, Any]] = []

    for msg in user_messages:
        text = msg["text"]
        low_text = text.lower()

        if len(text) < MIN_IMPORTANT_LEN:
            hit = any(kw.lower() in low_text for kw in IMPORTANT_KEYWORDS)
            if not hit:
                continue
        else:
            hit = any(kw.lower() in low_text for kw in IMPORTANT_KEYWORDS)
            if not hit:
                continue

        snippets.append(msg)

    return snippets


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_code_blocks_txt(path: str, blocks: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for idx, b in enumerate(blocks, start=1):
            f.write(f"===== CODE BLOCK {idx:06d} =====\n")
            f.write(f"Source: {b['source']}\n")
            lang = b.get("language") or ""
            if lang:
                f.write(f"Language: {lang}\n")
            f.write("\n")
            f.write(b["code"])
            f.write("\n\n")


def write_messages_txt(path: str, messages: List[Dict[str, Any]], title: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        for idx, m in enumerate(messages, start=1):
            f.write(f"===== MESSAGE {idx:06d} =====\n")
            f.write(f"Source: {m['source']}\n")
            f.write(f"Role:   {m.get('role', 'unknown')}\n\n")
            f.write(m["text"])
            f.write("\n\n")


# ---------------------------------------------------------------------------
# CORE PROCESSING FUNCTION
# ---------------------------------------------------------------------------

def process_zip(zip_path: str, out_dir: str, mode: str) -> None:
    out_dir = ensure_out_dir(out_dir)
    print(f"[INFO] ZIP path:   {zip_path}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Mode:       {mode}")

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    code_blocks: List[Dict[str, Any]] = []
    user_messages: List[Dict[str, Any]] = []
    other_messages: List[Dict[str, Any]] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir() and is_text_like(info.filename)]
        total_files = len(infos)
        print(f"[INFO] Text-like files in ZIP: {total_files}")

        pb = ProgressBar(total_steps=total_files if total_files > 0 else 1)

        for info in infos:
            filename = info.filename
            pb.advance(label=f"Processing {os.path.basename(filename)}")

            text = safe_read_text_from_zip(zf, info)
            if text is None:
                continue

            text = normalize_whitespace(text)
            source = filename

            if mode in ("all", "code"):
                blocks = extract_code_blocks_from_text(text, source)
                if blocks:
                    code_blocks.extend(blocks)

            if mode in ("all", "narrative", "knowledge"):
                handled = False
                if filename.lower().endswith(".json"):
                    try:
                        obj = json.loads(text)
                        json_walk_collect_messages(obj, source, user_messages, other_messages)
                        handled = True
                    except Exception:
                        handled = False

                if not handled:
                    text_transcript_split_roles(text, source, user_messages, other_messages)

        pb.done()

    # Deduplicate user messages by (source, text)
    seen = set()
    user_messages_dedup: List[Dict[str, Any]] = []
    for msg in user_messages:
        key = (msg["source"], msg["text"])
        if key not in seen:
            seen.add(key)
            user_messages_dedup.append(msg)
    user_messages = user_messages_dedup

    important_snippets: List[Dict[str, Any]] = []
    if mode in ("all", "knowledge"):
        important_snippets = build_important_snippets(user_messages)

    # Write outputs
    if mode in ("all", "code"):
        if code_blocks:
            code_txt_path = os.path.join(out_dir, "code_blocks_all.txt")
            code_json_path = os.path.join(out_dir, "code_blocks_all.json")
            write_code_blocks_txt(code_txt_path, code_blocks)
            write_json(code_json_path, code_blocks)
            print(f"[INFO] Wrote {len(code_blocks)} code blocks to:")
            print(f"       {code_txt_path}")
            print(f"       {code_json_path}")
        else:
            print("[INFO] No code blocks found.")

    if mode in ("all", "narrative", "knowledge"):
        if user_messages:
            narrative_txt_path = os.path.join(out_dir, "user_narrative_all.txt")
            narrative_json_path = os.path.join(out_dir, "user_narrative_all.json")
            write_messages_txt(
                narrative_txt_path,
                user_messages,
                title="User Narrative (All Conversations)",
            )
            write_json(narrative_json_path, user_messages)
            print(f"[INFO] Wrote {len(user_messages)} user messages to:")
            print(f"       {narrative_txt_path}")
            print(f"       {narrative_json_path}")
        else:
            print("[INFO] No user messages detected.")

    if mode in ("all", "knowledge"):
        if important_snippets:
            knowledge_txt_path = os.path.join(out_dir, "knowledge_snippets_all.txt")
            knowledge_json_path = os.path.join(out_dir, "knowledge_snippets_all.json")
            write_messages_txt(
                knowledge_txt_path,
                important_snippets,
                title="Important Knowledge Snippets (Auto-Selected)",
            )
            write_json(knowledge_json_path, important_snippets)
            print(f"[INFO] Wrote {len(important_snippets)} important snippets to:")
            print(f"       {knowledge_txt_path}")
            print(f"       {knowledge_json_path}")
        else:
            print("[INFO] No important snippets selected.")

    print("[INFO] Extraction complete.")


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    process_zip(ZIP_PATH, OUT_DIR, MODE)
'''

# ---------------------------
# litigation_os_nodes_compiler.py (D:\ as source, D:\LitigationOS\NODES as target)
# ---------------------------
nodes_compiler_code = r'''#!/usr/bin/env python3
"""
litigation_os_nodes_compiler.py

Purpose:
    Collect and normalize all "*nodes*" CSV/JSON/JSONL files into a single
    NODES directory for Litigation OS Diamond 9999+++.

    - Scans a source root (default: D:\\).
    - Finds files whose names contain "nodes" (case-insensitive) and whose
      extensions are .csv, .json, or .jsonl.
    - Converts them to canonical JSON lists under an output root
      (default: D:\\LitigationOS\\NODES).

    This lets the ingestion engine treat all "node" artifacts as JSON without
    you having to manually convert or move each file.

Defaults (for your system):

    SOURCE_ROOT = "D:\\"
    OUT_ROOT    = "D:\\LitigationOS\\NODES"

Usage (Windows, default roots):

    python litigation_os_nodes_compiler.py

Usage (override if ever needed):

    python litigation_os_nodes_compiler.py ^
      --source-root "D:\\GRAPH_ARTIFACTS" ^
      --out-root   "D:\\LitigationOS\\NODES"

You do not edit code; you only change command-line arguments if needed.
"""

import argparse
import json
import os
from typing import List, Tuple

import pandas as pd


DEFAULT_SOURCE_ROOT = r"D:\\"
DEFAULT_OUT_ROOT = r"D:\\LitigationOS\\NODES"


class ProgressBar:
    def __init__(self, total_steps: int, bar_width: int = 30) -> None:
        self.total_steps = max(1, total_steps)
        self.bar_width = bar_width
        self.current_step = 0

    def advance(self, label: str = "") -> None:
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps

        frac = self.current_step / self.total_steps
        filled = int(self.bar_width * frac)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        pct = int(frac * 100)
        print(f"\r[PROGRESS] [{bar}] {self.current_step}/{self.total_steps} {pct}% - {label}", end="", flush=True)

    def done(self) -> None:
        print()


def is_nodes_file(path: str) -> bool:
    name = os.path.basename(path).lower()
    if "nodes" not in name:
        return False
    ext = os.path.splitext(name)[1]
    return ext in (".csv", ".json", ".jsonl")


def collect_nodes_files(source_root: str, out_root: str) -> List[str]:
    """
    Walk source_root recursively and find node-like files.
    Skip anything already under out_root to avoid loops.
    """
    source_root = os.path.abspath(source_root)
    out_root = os.path.abspath(out_root)
    candidates: List[str] = []

    for dirpath, dirnames, filenames in os.walk(source_root):
        # Skip output root subtree
        if os.path.abspath(dirpath).startswith(out_root):
            continue
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if is_nodes_file(full):
                candidates.append(full)

    candidates.sort()
    return candidates


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dest_path_for(source_path: str, out_root: str) -> str:
    """
    Map D:\\something\\nodes_foo.csv -> out_root\\nodes_foo.json.
    Ignore subdir structure to keep paths simple.
    """
    base_name = os.path.basename(source_path)
    name, _ext = os.path.splitext(base_name)
    return os.path.join(out_root, f"{name}.json")


def convert_csv_to_json(src: str, dst: str) -> None:
    df = pd.read_csv(src)
    records = df.to_dict(orient="records")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def convert_json_to_json(src: str, dst: str) -> None:
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    # If top-level is not list, wrap it to keep ingestion tolerant.
    if not isinstance(data, list):
        data = [data]
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def convert_jsonl_to_json(src: str, dst: str) -> None:
    records = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            records.append(obj)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def compile_nodes(source_root: str, out_root: str) -> Tuple[int, int]:
    """
    Return (converted_count, skipped_count).
    """
    source_root = os.path.abspath(source_root)
    out_root = os.path.abspath(out_root)
    ensure_dir(out_root)

    candidates = collect_nodes_files(source_root, out_root)
    print(f"[NODES] Source root: {source_root}")
    print(f"[NODES] Out root:    {out_root}")
    print(f"[NODES] Candidate node files discovered: {len(candidates)}")

    pb = ProgressBar(total_steps=len(candidates) if candidates else 1)
    converted = 0
    skipped = 0

    for src in candidates:
        ext = os.path.splitext(src)[1].lower()
        dst = dest_path_for(src, out_root)
        label = os.path.basename(src)
        pb.advance(label=f"Converting {label}")

        try:
            if ext == ".csv":
                convert_csv_to_json(src, dst)
            elif ext == ".json":
                convert_json_to_json(src, dst)
            elif ext == ".jsonl":
                convert_jsonl_to_json(src, dst)
            else:
                skipped += 1
                continue
            converted += 1
        except Exception as exc:
            skipped += 1
            print(f"\n[WARN] Failed to convert {src}: {exc}")

    pb.done()
    return converted, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Litigation OS NODES compiler")
    parser.add_argument(
        "--source-root",
        default=DEFAULT_SOURCE_ROOT,
        help=f"Root to scan for *nodes* files (default: {DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT,
        help=f"Output NODES directory (default: {DEFAULT_OUT_ROOT})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    converted, skipped = compile_nodes(args.source_root, args.out_root)
    print(f"[NODES] Conversion complete. Converted={converted}, skipped={skipped}")
'''

# ---------------------------
# litigation_os_master.py (master orchestrator: nodes compiler + pipeline, with progress)
# ---------------------------
master_code = r'''#!/usr/bin/env python3
"""
litigation_os_master.py

Master orchestrator for Litigation OS • Diamond 9999+++ on Windows.

Sequence (one command):

    1. Compile all *nodes* CSV/JSON/JSONL on D:\ into D:\LitigationOS\NODES.
    2. Run the full Litigation OS pipeline (DB + scores + topics + dashboards + graph).

This script stitches together:

    - litigation_os_nodes_compiler.compile_nodes(...)
    - litigation_os_launcher.detect_defaults()
    - litigation_os_orchestrator.run_full_pipeline(...)

You run this, not the individual pieces, when you just want
"everything important" executed in order.

Usage:

    cd /d D:\LitigationOS_SCRIPTS_Diamond
    python litigation_os_master.py

or double-click the BAT:

    run_litigation_os_master_default.bat
"""

from __future__ import annotations

import os

from litigation_os_nodes_compiler import compile_nodes, DEFAULT_SOURCE_ROOT, DEFAULT_OUT_ROOT
import litigation_os_launcher as launcher
from litigation_os_orchestrator import run_full_pipeline


class ProgressBar:
    def __init__(self, total_steps: int, bar_width: int = 30) -> None:
        self.total_steps = max(1, total_steps)
        self.bar_width = bar_width
        self.current_step = 0

    def advance(self, label: str = "") -> None:
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps

        frac = self.current_step / self.total_steps
        filled = int(self.bar_width * frac)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        pct = int(frac * 100)
        print(f"\r[MASTER] [{bar}] {self.current_step}/{self.total_steps} {pct}% - {label}", end="", flush=True)

    def done(self) -> None:
        print()


def main() -> None:
    print("[MASTER] Litigation OS • Diamond 9999+++ • master run")

    pb = ProgressBar(total_steps=2)

    # ------------------------------------------------------------------
    # 1. Compile nodes into D:\LitigationOS\NODES
    # ------------------------------------------------------------------
    pb.advance("Compiling *nodes* files into D:\\LitigationOS\\NODES")
    converted, skipped = compile_nodes(DEFAULT_SOURCE_ROOT, DEFAULT_OUT_ROOT)
    print(f"\n[MASTER] Nodes compiler: converted={converted}, skipped={skipped}")

    # ------------------------------------------------------------------
    # 2. Run full pipeline using launcher defaults
    # ------------------------------------------------------------------
    data_root, db_path, dash_dir, graph_dir = launcher.detect_defaults()
    label = f"Running full pipeline (data_root={data_root})"
    pb.advance(label)
    run_full_pipeline(
        data_root=data_root,
        db_path=db_path,
        dashboard_dir=dash_dir,
        graph_dir=graph_dir,
        text_glob="**/*.json",
        min_topic_docs=10,
        n_topics=12,
    )

    pb.done()
    print("[MASTER] Master run complete.")


if __name__ == "__main__":
    main()
'''

# ---------------------------
# README (D:\ layout, with master orchestrator)
# ---------------------------
readme_text = """Litigation OS • Diamond 9999+++ • Core Scripts Bundle (D:)
==============================================================

This ZIP contains the core Python scripts and helpers for your upgraded
Litigation OS pipeline (Diamond 9999+++), wired to D:\\ as the canonical root:

  - litigation_os_advanced_engines.py
  - litigation_os_orchestrator.py
  - litigation_os_launcher.py
  - litigation_os_nodes_compiler.py
  - litigation_os_master.py
  - litigation_os_chat_extractor_local.py
  - run_litigation_os_default.bat
  - run_litigation_os_master_default.bat
  - run_nodes_compiler_default.bat
  - run_chat_extractor_default.bat
  - install_litigation_os_deps.bat
  - README_LitigationOS_Diamond_9999+++.txt

------------------------------------------------------------------
1. Scripts folder on D:
------------------------------------------------------------------

Use a fixed scripts folder:

    D:\\LitigationOS_SCRIPTS_Diamond\\

Unzip this bundle into that folder. You do not rename or edit files.

------------------------------------------------------------------
2. Canonical Litigation OS layout on D:
------------------------------------------------------------------

The code defaults to:

    NODES:        D:\\LitigationOS\\NODES
    SQLite DB:    D:\\LitigationOS\\litigation_os.db
    Dashboards:   D:\\LitigationOS\\DASHBOARDS
    Graph output: D:\\LitigationOS\\GRAPHS\\ADVANCED

You do not have to create these folders manually; the engines will create them
as needed. Only ensure the base root exists:

    D:\\LitigationOS\\

------------------------------------------------------------------
3. Installing Python dependencies (one time)
------------------------------------------------------------------

From Command Prompt:

    cd /d D:\\LitigationOS_SCRIPTS_Diamond
    install_litigation_os_deps.bat

This runs:

    python -m pip install --upgrade pip
    pip install numpy pandas plotly scikit-learn

If python or pip are not in PATH, that is a one-time OS configuration step.

------------------------------------------------------------------
4. Building the NODES directory automatically (from D:\\)
------------------------------------------------------------------

The system expects JSON manifests under:

    D:\\LitigationOS\\NODES

You may already have many *nodes* CSV/JSON/JSONL files on D:\\
(e.g. mcr_nodes.csv, unified_nodes.json, MindEye2_nodes.json, etc).

The nodes compiler handles them:

    cd /d D:\\LitigationOS_SCRIPTS_Diamond
    run_nodes_compiler_default.bat

What it does:

  - Recursively scans D:\\ for files whose names contain "nodes".
  - Accepts extensions: .csv, .json, .jsonl
  - Converts each to a JSON list:
      * CSV   -> list of dicts (via pandas)
      * JSON  -> wrapped into a list if needed
      * JSONL -> one JSON object per line -> list
  - Writes them as .json into:

        D:\\LitigationOS\\NODES\\

After this, D:\\LitigationOS\\NODES is the unified ingest root.

------------------------------------------------------------------
5. Running the full Litigation OS pipeline (single-step master)
------------------------------------------------------------------

The master orchestrator sequences everything for you:

    1. Compile nodes into D:\\LitigationOS\\NODES
    2. Run the full Litigation OS pipeline (DB + scores + topics + dashboards + graph)

Recommended:

  A. Double-click:

        run_litigation_os_master_default.bat

  B. Or from Command Prompt:

        cd /d D:\\LitigationOS_SCRIPTS_Diamond
        python litigation_os_master.py

Internally this calls:

  - litigation_os_nodes_compiler.compile_nodes(...)
  - litigation_os_orchestrator.run_full_pipeline(...)

using the D:\\LitigationOS defaults.

------------------------------------------------------------------
6. Running pieces individually (optional)
------------------------------------------------------------------

If you want to run just parts:

A) Only compile nodes:

    cd /d D:\\LitigationOS_SCRIPTS_Diamond
    run_nodes_compiler_default.bat

B) Only run the main pipeline (expects D:\\LitigationOS\\NODES populated):

    cd /d D:\\LitigationOS_SCRIPTS_Diamond
    run_litigation_os_default.bat

This uses:

    data_root    = D:\\LitigationOS\\NODES
    db_path      = D:\\LitigationOS\\litigation_os.db
    dashboardDir = D:\\LitigationOS\\DASHBOARDS
    graphDir     = D:\\LitigationOS\\GRAPHS\\ADVANCED

You can override paths via CLI flags to litigation_os_launcher.py if desired.

------------------------------------------------------------------
7. Running the ChatGPT export extractor (separate tool)
------------------------------------------------------------------

The chat extractor is independent from the main Litigation OS pipeline.
It is wired to your current ChatGPT export ZIP:

    C:\\Users\\andre\\Downloads\\5a196c9....zip

It produces:

    C:\\Users\\andre\\Downloads\\chat_extract_out\\
        code_blocks_all.txt / .json
        user_narrative_all.txt / .json
        knowledge_snippets_all.txt / .json

To run:

  A. Double-click:

        run_chat_extractor_default.bat

  B. Or from Command Prompt:

        cd /d D:\\LitigationOS_SCRIPTS_Diamond
        python litigation_os_chat_extractor_local.py

If you later download a new export ZIP with a different name, either rename
the ZIP to match the hard-wired path or update ZIP_PATH in the script.

------------------------------------------------------------------
8. What “activated” looks like
------------------------------------------------------------------

After a successful master run (litigation_os_master.py), you should see:

  - D:\\LitigationOS\\litigation_os.db
  - D:\\LitigationOS\\NODES\\... (compiled JSON nodes)
  - D:\\LitigationOS\\DASHBOARDS\\events_timeline.html
  - D:\\LitigationOS\\DASHBOARDS\\time_series_heatmap.html
  - D:\\LitigationOS\\GRAPHS\\ADVANCED\\nodes_advanced.json
  - D:\\LitigationOS\\GRAPHS\\ADVANCED\\edges_advanced.json

Console output ends with:

    [MASTER] Master run complete.

At that point, Litigation OS • Diamond 9999+++ is fully live on D:.
"""

# ---------------------------
# BAT launchers
# ---------------------------
run_pipeline_bat = r"""@echo off
REM Run only the Litigation OS pipeline (expects NODES already compiled).
cd /d "%~dp0"
python litigation_os_launcher.py
pause
"""

run_master_bat = r"""@echo off
REM Run the master orchestrator (nodes compiler + full pipeline).
cd /d "%~dp0"
python litigation_os_master.py
pause
"""

run_nodes_bat = r"""@echo off
REM Compile all *nodes* CSV/JSON/JSONL under D:\ into D:\LitigationOS\NODES.
cd /d "%~dp0"
python litigation_os_nodes_compiler.py
pause
"""

run_chat_bat = r"""@echo off
REM Run the ChatGPT export extractor with the hard-wired ZIP path.
cd /d "%~dp0"
python litigation_os_chat_extractor_local.py
pause
"""

install_deps_bat = r"""@echo off
REM Install core Python dependencies for Litigation OS Diamond 9999+++.
python -m pip install --upgrade pip
pip install numpy pandas plotly scikit-learn
pause
"""

# Write all files into root
files = {
    "litigation_os_advanced_engines.py": advanced_engines_code,
    "litigation_os_orchestrator.py": orchestrator_code,
    "litigation_os_launcher.py": launcher_code,
    "litigation_os_chat_extractor_local.py": chat_extractor_code,
    "litigation_os_nodes_compiler.py": nodes_compiler_code,
    "litigation_os_master.py": master_code,
    "README_LitigationOS_Diamond_9999+++.txt": readme_text,
    "run_litigation_os_default.bat": run_pipeline_bat,
    "run_litigation_os_master_default.bat": run_master_bat,
    "run_nodes_compiler_default.bat": run_nodes_bat,
    "run_chat_extractor_default.bat": run_chat_bat,
    "install_litigation_os_deps.bat": install_deps_bat,
}

for name, content in files.items():
    with open(os.path.join(root, name), "w", encoding="utf-8") as f:
        f.write(content)

# Create ZIP
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
zip_name = f"LitigationOS_Diamond_9999+++_Core_D_{timestamp}.zip"
zip_path = os.path.join("/mnt/data", zip_name)

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for name in files.keys():
        z.write(os.path.join(root, name), arcname=name)

zip_path

import datetime
import os
import zipfile

root = "/mnt/data/LitigationOS_Diamond_D_Patch"
os.makedirs(root, exist_ok=True)

launcher_code = r"""#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Tuple
from litigation_os_orchestrator import run_full_pipeline

def detect_defaults() -> Tuple[str, str, str, str]:
    is_windows = os.name == "nt"
    if is_windows:
        base = "D:/LitigationOS"
    else:
        base = "/storage/emulated/0/Download/LitigationOS"
    data_root = os.path.join(base, "NODES")
    db_path = os.path.join(base, "litigation_os.db")
    dashboard_dir = os.path.join(base, "DASHBOARDS")
    graph_dir = os.path.join(base, "GRAPHS", "ADVANCED")
    return data_root, db_path, dashboard_dir, graph_dir

def parse_args() -> argparse.Namespace:
    d_data_root, d_db_path, d_dash_dir, d_graph_dir = detect_defaults()
    p = argparse.ArgumentParser(description="Litigation OS launcher (D: defaults)")
    p.add_argument("--data-root", default=d_data_root)
    p.add_argument("--db-path", default=d_db_path)
    p.add_argument("--dashboard-dir", default=d_dash_dir)
    p.add_argument("--graph-dir", default=d_graph_dir)
    p.add_argument("--text-glob", default="**/*.json")
    p.add_argument("--min-topic-docs", type=int, default=10)
    p.add_argument("--n-topics", type=int, default=12)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    print("[LAUNCH] Litigation OS full pipeline")
    print(f"[LAUNCH] data_root    = {args.data_root}")
    print(f"[LAUNCH] db_path      = {args.db_path}")
    print(f"[LAUNCH] dashboardDir = {args.dashboard_dir}")
    print(f"[LAUNCH] graphDir     = {args.graph_dir}")
    sys.stdout.flush()
    run_full_pipeline(
        data_root=args.data_root,
        db_path=args.db_path,
        dashboard_dir=args.dashboard_dir,
        graph_dir=args.graph_dir,
        text_glob=args.text_glob,
        min_topic_docs=args.min_topic_docs,
        n_topics=args.n_topics,
    )
    print("[LAUNCH] Pipeline finished.")

if __name__ == "__main__":
    main()
"""

nodes_code = r"""#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Tuple
import pandas as pd

DEFAULT_SOURCE_ROOT = r"D:\\"
DEFAULT_OUT_ROOT = r"D:\\LitigationOS\\NODES"

class ProgressBar:
    def __init__(self, total_steps: int, bar_width: int = 30) -> None:
        self.total_steps = max(1, total_steps)
        self.bar_width = bar_width
        self.current_step = 0
    def advance(self, label: str = "") -> None:
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps
        frac = self.current_step / self.total_steps
        filled = int(self.bar_width * frac)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        pct = int(frac * 100)
        print(f"\r[NODES] [{bar}] {self.current_step}/{self.total_steps} {pct}% - {label}", end="", flush=True)
    def done(self) -> None:
        print()

def is_nodes_file(path: str) -> bool:
    name = os.path.basename(path).lower()
    if "nodes" not in name:
        return False
    ext = os.path.splitext(name)[1].lower()
    return ext in (".csv", ".json", ".jsonl")

def collect_nodes_files(source_root: str, out_root: str) -> List[str]:
    source_root = os.path.abspath(source_root)
    out_root = os.path.abspath(out_root)
    out_root_norm = os.path.join(out_root, "")
    out_files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(source_root):
        if os.path.abspath(dirpath).startswith(out_root_norm):
            continue
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if is_nodes_file(full):
                out_files.append(full)
    out_files.sort()
    return out_files

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def dest_path_for(source_path: str, out_root: str) -> str:
    base = os.path.basename(source_path)
    name, _ = os.path.splitext(base)
    return os.path.join(out_root, f"{name}.json")

def convert_csv_to_json(src: str, dst: str) -> None:
    df = pd.read_csv(src)
    records = df.to_dict(orient="records")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def convert_json_to_json(src: str, dst: str) -> None:
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def convert_jsonl_to_json(src: str, dst: str) -> None:
    records = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            records.append(obj)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def compile_nodes(source_root: str, out_root: str) -> Tuple[int, int]:
    source_root = os.path.abspath(source_root)
    out_root = os.path.abspath(out_root)
    ensure_dir(out_root)
    candidates = collect_nodes_files(source_root, out_root)
    print(f"[NODES] Source root: {source_root}")
    print(f"[NODES] Out root:    {out_root}")
    print(f"[NODES] Candidate node files: {len(candidates)}")
    pb = ProgressBar(total_steps=len(candidates) if candidates else 1)
    converted = 0
    skipped = 0
    for src in candidates:
        ext = os.path.splitext(src)[1].lower()
        dst = dest_path_for(src, out_root)
        label = os.path.basename(src)
        pb.advance(label=f"Converting {label}")
        try:
            if ext == ".csv":
                convert_csv_to_json(src, dst)
            elif ext == ".json":
                convert_json_to_json(src, dst)
            elif ext == ".jsonl":
                convert_jsonl_to_json(src, dst)
            else:
                skipped += 1
                continue
            converted += 1
        except Exception as exc:
            skipped += 1
            print(f"\n[WARN] Failed to convert {src}: {exc}")
    pb.done()
    return converted, skipped

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Litigation OS NODES compiler (D:)")
    p.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    conv, skip = compile_nodes(args.source_root, args.out_root)
    print(f"[NODES] Conversion complete. Converted={conv}, skipped={skip}")
"""

master_code = r"""#!/usr/bin/env python3
from litigation_os_nodes_compiler import compile_nodes, DEFAULT_SOURCE_ROOT, DEFAULT_OUT_ROOT
import litigation_os_launcher as launcher
from litigation_os_orchestrator import run_full_pipeline

class ProgressBar:
    def __init__(self, total_steps: int, bar_width: int = 30) -> None:
        self.total_steps = max(1, total_steps)
        self.bar_width = bar_width
        self.current_step = 0
    def advance(self, label: str = "") -> None:
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps
        frac = self.current_step / self.total_steps
        filled = int(self.bar_width * frac)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        pct = int(frac * 100)
        print(f"\r[MASTER] [{bar}] {self.current_step}/{self.total_steps} {pct}% - {label}", end="", flush=True)
    def done(self) -> None:
        print()

def main() -> None:
    print("[MASTER] Litigation OS • Diamond 9999+++ master run (D:)")
    pb = ProgressBar(total_steps=2)
    pb.advance("Compiling *nodes* into D:\\LitigationOS\\NODES")
    conv, skip = compile_nodes(DEFAULT_SOURCE_ROOT, DEFAULT_OUT_ROOT)
    print(f"\n[MASTER] Nodes compiler: converted={conv}, skipped={skip}")
    data_root, db_path, dash_dir, graph_dir = launcher.detect_defaults()
    label = f"Running full pipeline (data_root={data_root})"
    pb.advance(label)
    run_full_pipeline(
        data_root=data_root,
        db_path=db_path,
        dashboard_dir=dash_dir,
        graph_dir=graph_dir,
        text_glob="**/*.json",
        min_topic_docs=10,
        n_topics=12,
    )
    pb.done()
    print("[MASTER] Master run complete.")

if __name__ == "__main__":
    main()
"""

readme_patch = """Litigation OS • Diamond 9999+++ • D: drive patch
==============================================

This ZIP only contains the drive-specific launcher + nodes compiler +
master orchestrator and BAT files for running everything in sequence
on D:.

Drop these into your existing scripts folder:

    D:\\LitigationOS_SCRIPTS_Diamond\\

and allow overwrite.

Included:

  - litigation_os_launcher.py      (defaults to D:\\LitigationOS)
  - litigation_os_nodes_compiler.py (scans D:\\, writes D:\\LitigationOS\\NODES)
  - litigation_os_master.py        (runs compiler + full pipeline)
  - run_litigation_os_default.bat
  - run_litigation_os_master_default.bat
  - run_nodes_compiler_default.bat

Usage (preferred, one step):

    cd /d D:\\LitigationOS_SCRIPTS_Diamond
    run_litigation_os_master_default.bat

This will:

  1) Compile all *nodes* CSV/JSON/JSONL on D:\\ into D:\\LitigationOS\\NODES
  2) Run the full Litigation OS pipeline (DB + scores + topics + dashboards + graph)
"""

run_pipeline_bat = r"""@echo off
cd /d "%~dp0"
python litigation_os_launcher.py
pause
"""

run_master_bat = r"""@echo off
cd /d "%~dp0"
python litigation_os_master.py
pause
"""

run_nodes_bat = r"""@echo off
cd /d "%~dp0"
python litigation_os_nodes_compiler.py
pause
"""

files = {
    "litigation_os_launcher.py": launcher_code,
    "litigation_os_nodes_compiler.py": nodes_code,
    "litigation_os_master.py": master_code,
    "README_D_Patch.txt": readme_patch,
    "run_litigation_os_default.bat": run_pipeline_bat,
    "run_litigation_os_master_default.bat": run_master_bat,
    "run_nodes_compiler_default.bat": run_nodes_bat,
}

for name, content in files.items():
    with open(os.path.join(root, name), "w", encoding="utf-8") as f:
        f.write(content)

zip_name = "LitigationOS_Diamond_D_Patch.zip"
zip_path = os.path.join("/mnt/data", zip_name)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for name in files.keys():
        z.write(os.path.join(root, name), arcname=name)

zip_path
