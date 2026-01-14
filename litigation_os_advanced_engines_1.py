import datetime
import os
import pathlib
import textwrap
import zipfile

root = "/mnt/data/LitigationOS_Diamond_9999+++_Core"
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
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Litigation OS Orchestrator with progress bar")
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root containing JSON node/exhibit manifests "
             "(e.g. F:/LitigationOS/NODES or D:/.../NODES)",
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="SQLite DB path, e.g. F:/LitigationOS/litigation_os.db",
    )
    parser.add_argument(
        "--dashboard-dir",
        required=True,
        help="Folder for Plotly HTML dashboards (e.g. F:/LitigationOS/DASHBOARDS)",
    )
    parser.add_argument(
        "--graph-dir",
        required=True,
        help="Folder for graph JSON outputs (e.g. F:/LitigationOS/GRAPHS/ADVANCED)",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
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
# litigation_os_launcher.py
# ---------------------------
launcher_code = r'''#!/usr/bin/env python3
"""
litigation_os_launcher.py

One-command launcher for the upgraded Litigation OS pipeline.

Stack:
- litigation_os_advanced_engines.py
- litigation_os_orchestrator.py

On Windows, defaults assume:
    F:/LitigationOS/NODES
    F:/LitigationOS/litigation_os.db
    F:/LitigationOS/DASHBOARDS
    F:/LitigationOS/GRAPHS/ADVANCED

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
        base = "F:/LitigationOS"
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
# Remaining large code sections truncated for stability.
