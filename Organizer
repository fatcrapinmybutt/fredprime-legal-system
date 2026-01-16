function myFunction(/**
 * DRIVE BUCKETIZER + DEDUPE + AGGRESSIVE EMPTY-FOLDER PRUNE + LIVE DASHBOARD (Final v4.2, merged fixes)
 * ---------------------------------------------------------------------------------------------------
 * Scope: My Drive only (DriveApp.getRootFolder()).
 * Shortcuts: skipped (no moves, no dedupe actions).
 * BUCKETS/: created under My Drive if missing; never reorganized; never pruned.
 * ZIPs: go to BUCKETS/ZIPS/ (no extraction).
 * Dedupe: within BUCKETS only; MD5 when available; else name+size+mimeType fallback;
 *         keep oldest modified; quarantine duplicates to BUCKETS/_DEDUP_QUARANTINE/.
 * Prune: aggressive recursive prune of empty folders across My Drive, excluding:
 *        - BUCKETS subtree
 *        - denylisted top-level roots (by folder name) AND their descendants
 *
 * Resumable: Script Properties state (queues + page tokens + prune stack).
 * Dashboard: live summary in BUCKETS_AUDIT_LOG spreadsheet tab "dashboard" updated every run.
 *
 * REQUIRED ONE-TIME SETUP:
 * 1) Apps Script: Services -> enable "Drive API" (Advanced Google Services)
 * 2) Google Cloud project: enable "Google Drive API"
 *
 * Recommended workflow:
 * - selfTest() once
 * - dryRunOnce()
 * - runNow()
 * - installTrigger()
 */

// ========================= CONFIG =========================

 CFG = {
  BUCKETS_DIR_NAME: "BUCKETS",

  LOG_SPREADSHEET_NAME: "BUCKETS_AUDIT_LOG",
  LOG_SHEET_TAB: "log",
  LOG_MAX_ROWS_SOFT: 900000,

  // Never reorganize/prune these top-level roots (by exact folder name).
  // Their descendants are protected as well.
  DENYLIST_TOPLEVEL_FOLDER_NAMES: [
    "LITIGATION_OS$",
    "EDS-USB",
    "NODES",
    "BUCKETS",
    "USB and External Devices",
    "Colab Notebooks"
  ],

  // Work budgets per run
  SOFT_TIME_LIMIT_MS: 5 * 60 * 1000,
  MAX_FILES_PER_RUN: 800,
  MAX_FOLDERS_PER_RUN: 300,

  // Dedupe scan budget (within BUCKETS only)
  MAX_DEDUPE_FILES_PER_RUN: 1200,
  MAX_DEDUPE_FOLDERS_PER_RUN: 400,

  // Aggressive prune budget
  ENABLE_EMPTY_FOLDER_PRUNING: true,
  MAX_PRUNE_FOLDER_CHECKS_PER_RUN: 1200,
  MAX_PRUNE_TRASH_PER_RUN: 400,

  // Collision policy
  COLLISION_POLICY: "suffix", // "suffix" | "timestamp"

  // Dedupe defaults
  ENABLE_DEDUPE: true,
  DEDUPE_QUARANTINE_BUCKET: "_DEDUP_QUARANTINE",

  // Persistent bounded dedupe index (speeds dedupe across multiple runs)
  ENABLE_PERSISTENT_DEDUPE_INDEX: true,
  DEDUPE_INDEX_MAX_KEYS: 15000,
  DEDUPE_INDEX_PROP_KEY: "BUCKETIZER_DEDUPE_INDEX_V1",

  // Cache tuning
  MAX_PARENT_CACHE_ENTRIES: 50000, // soft bound (best-effort; avoids unbounded growth)
  MAX_META_CACHE_ENTRIES: 50000,

  // Bucket map (extension-based)
  BUCKETS: {
    "PDFS": [".pdf"],
    "DOCS": [".doc", ".docx", ".odt", ".rtf", ".txt", ".md"],
    "SHEETS": [".xls", ".xlsx", ".ods", ".csv", ".tsv"],
    "SLIDES": [".ppt", ".pptx", ".odp"],
    "IMAGES": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic"],
    "AUDIO": [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"],
    "VIDEO": [".mp4", ".mov", ".mkv", ".avi", ".wmv", ".webm"],
    "ZIPS": [".zip"],                       // ZIPs bucket (no extraction)
    "ARCHIVES_OTHER": [".rar", ".7z", ".tar", ".gz", ".tgz", ".iso"],
    "SCRIPTS": [".py", ".ps1", ".sh", ".bat", ".cmd", ".js", ".ts", ".java", ".c", ".cpp", ".html", ".htm", ".css", ".json", ".sql"],
    "CONFIG": [".yml", ".yaml", ".ini", ".cfg", ".conf", ".toml", ".lock"],
    "OTHER": []
  },

  // Google-native MIME buckets
  MIME_BUCKETS: {
    "application/vnd.google-apps.document": "DOCS",
    "application/vnd.google-apps.spreadsheet": "SHEETS",
    "application/vnd.google-apps.presentation": "SLIDES",
    "application/vnd.google-apps.drawing": "IMAGES",
    "application/vnd.google-apps.form": "OTHER"
  },

  // Backoff tuning
  RETRY_MAX_ATTEMPTS: 6,
  RETRY_BASE_SLEEP_MS: 250,
  RETRY_JITTER_MS: 200
}

// ========================= DASHBOARD CONFIG =========================

 = {
  TAB_NAME: "dashboard",
  KV_START_ROW: 2
};

// ========================= STATE MODEL =========================
// Stored in Script Properties under key BUCKETIZER_STATE_V4_2
// {
//   initialized: true|false,
//   stateVersion: "4.2",
//
//   folderQueue: [folderId...],
//   currentFolderId: "id"|null,
//   currentPageToken: "..."|null,
//
//   dedupeQueue: [folderId...],
//   dedupeCurrentFolderId: "id"|null,
//   dedupeCurrentPageToken: "..."|null,
//
//   pruneStack: [ {id, expanded} ... ],
//   pruneRootsSeeded: true|false
// }

// ========================= ENTRYPOINTS =========================

function selfTest() {
  const ctx = initContext_(Date.now(), true);

  // Drive API sanity: list 1 item from root
  const r = callDriveList_(() => Drive.Files.list({
    q: `'${ctx.rootId}' in parents and trashed = false`,
    fields: "files(id,name,mimeType), nextPageToken",
    pageSize: 1,
    supportsAllDrives: false
  }));
  Logger.log("SELFTEST root list ok: " + JSON.stringify(r.files || []));

  // Sheet write sanity
  ctx.logBuffer.push(logObj_("selftest_ok", { runId: ctx.runId, rootId: ctx.rootId, bucketsId: ctx.bucketsRootId }));
  flushLogs_(ctx, { runId: ctx.runId, selftest: true });
  updateDashboard_(ctx, loadState_(), { runId: ctx.runId, selftest: true, dryRun: true });
}

function dryRunOnce() {
  const ctx = initContext_(Date.now(), true);
  const budget = runPass_(ctx);
  finalizeRun_(ctx, budget);
}

function runNow() {
  const ctx = initContext_(Date.now(), false);
  const budget = runPass_(ctx);
  finalizeRun_(ctx, budget);
}

function installTrigger() {
  const triggers = ScriptApp.getProjectTriggers();
  for (const t of triggers) {
    if (t.getHandlerFunction() === "runNow") ScriptApp.deleteTrigger(t);
  }
  ScriptApp.newTrigger("runNow").timeBased().everyMinutes(15).create();
  Logger.log("Trigger installed: runNow every 15 minutes");
}

function resetState() {
  PropertiesService.getScriptProperties().deleteProperty("BUCKETIZER_STATE_V4_2");
  Logger.log("State reset: BUCKETIZER_STATE_V4_2 deleted");
}

function resetDedupeIndex() {
  PropertiesService.getScriptProperties().deleteProperty(CFG.DEDUPE_INDEX_PROP_KEY);
  Logger.log("Dedupe index reset: " + CFG.DEDUPE_INDEX_PROP_KEY + " deleted");
}

// ========================= CORE PASS =========================

function runPass_(ctx) {
  const budget = {
    runId: ctx.runId,
    dryRun: ctx.dryRun,
    startedIso: new Date(ctx.startedMs).toISOString(),

    filesProcessed: 0,
    foldersProcessed: 0,
    moved: 0,
    collisionsRenamed: 0,
    skipped: 0,
    errors: 0,

    dedupeFilesScanned: 0,
    dedupeFoldersScanned: 0,
    dedupeCandidates: 0,
    dedupeQuarantined: 0,
    dedupeRenamed: 0,
    dedupeErrors: 0,
    dedupeIndexHits: 0,
    dedupeIndexAdds: 0,
    dedupeIndexEvicts: 0,

    pruneFolderChecks: 0,
    prunedFoldersTrashed: 0,
    prunedErrors: 0
  };

  const state = loadState_();

  if (!state.initialized) {
    initializeState_(ctx, state);
    ctx.logBuffer.push(logObj_("state_init", { runId: ctx.runId, topLevelSeedCount: state.folderQueue.length }));
  } else if (!state.folderQueue || state.folderQueue.length < 10) {
    // Best-effort reseed
    const extra = seedTopLevelFolders_(ctx);
    const seen = toSet_(state.folderQueue || []);
    for (const id of extra) if (!seen[id]) state.folderQueue.push(id);
    saveState_(state);
    ctx.logBuffer.push(logObj_("state_reseed_top", { runId: ctx.runId, added: extra.length }));
  }

  // 1) BUCKETIZATION
  bucketizeTraversal_(ctx, state, budget);

  // 2) DEDUPE within BUCKETS
  if (CFG.ENABLE_DEDUPE && timeLeft_(ctx)) {
    dedupeTraversal_(ctx, state, budget);
  }

  // 3) AGGRESSIVE PRUNE empty folders recursively
  if (CFG.ENABLE_EMPTY_FOLDER_PRUNING && timeLeft_(ctx)) {
    aggressivePruneEmptyFolders_(ctx, state, budget);
  }

  // Live dashboard update (every run)
  try {
    updateDashboard_(ctx, state, budget);
  } catch (e) {
    ctx.logBuffer.push(logObj_("error", { runId: ctx.runId, stage: "dashboard_update", reason: String(e) }));
  }

  saveState_(state);
  return budget;
}

function initializeState_(ctx, state) {
  const seed = seedTopLevelFolders_(ctx);

  // IMPORTANT FIX: include root so loose files directly in My Drive root are bucketized
  state.folderQueue = [ctx.rootId].concat(seed);

  state.currentFolderId = null;
  state.currentPageToken = null;

  state.dedupeQueue = [ctx.bucketsRootId];
  state.dedupeCurrentFolderId = null;
  state.dedupeCurrentPageToken = null;

  state.pruneStack = [];
  state.pruneRootsSeeded = false;

  state.initialized = true;
  state.stateVersion = "4.2";

  saveState_(state);
}

// ========================= CONTEXT INIT =========================

function initContext_(startedMs, dryRun) {
  const runId = "RUN_" + Utilities.getUuid();

  const log = ensureLogSheet_();

  // Ensure dashboard exists early
  try { ensureDashboardSheet_(log.spreadsheetId); } catch (e) { Logger.log("DASHBOARD_INIT_FAIL " + String(e)); }

  const root = DriveApp.getRootFolder();
  const rootId = root.getId();

  const bucketsRootId = ensureFolderUnderParent_(rootId, CFG.BUCKETS_DIR_NAME);

  const bucketIds = {};
  for (const b of Object.keys(CFG.BUCKETS)) bucketIds[b] = ensureFolderUnderParent_(bucketsRootId, b);
  bucketIds[CFG.DEDUPE_QUARANTINE_BUCKET] = ensureFolderUnderParent_(bucketsRootId, CFG.DEDUPE_QUARANTINE_BUCKET);

  const denyNameSet = toSet_(CFG.DENYLIST_TOPLEVEL_FOLDER_NAMES);

  return {
    runId,
    startedMs,
    dryRun,
    logSpreadsheetId: log.spreadsheetId,
    logTabName: log.tabName,

    rootId,
    bucketsRootId,
    bucketIds,
    denyNameSet,

    // caches
    destNameCache: {},
    folderUnderBucketsCache: {},
    folderUnderDenyRootCache: {},
    parentOfFolderCache: {},     // folderId -> parentId|null
    metaCache: {},               // id -> {id,parents,mimeType,name,modifiedTime,size,md5Checksum}

    // best-effort guards
    _denyTopIds: null,

    logBuffer: []
  };
}

// ========================= BUCKETIZATION =========================

function bucketizeTraversal_(ctx, state, budget) {
  while (timeLeft_(ctx)) {
    if (budget.filesProcessed >= CFG.MAX_FILES_PER_RUN) break;
    if (budget.foldersProcessed >= CFG.MAX_FOLDERS_PER_RUN) break;

    if (!state.currentFolderId) {
      if (!state.folderQueue || state.folderQueue.length === 0) break;
      state.currentFolderId = state.folderQueue.shift();
      state.currentPageToken = null;
      saveState_(state);
    }

    const folderId = state.currentFolderId;

    // Never prune/skip root for bucketization (root should be processed for loose files)
    // But do skip BUCKETS subtree + denylisted subtree
    if (folderId !== ctx.rootId) {
      if (folderId === ctx.bucketsRootId || folderIsUnderBuckets_(ctx, folderId) || folderIsUnderDenylistedRoot_(ctx, folderId)) {
        state.currentFolderId = null;
        state.currentPageToken = null;
        continue;
      }
    }

    const page = listFilesInFolder_(ctx, folderId, state.currentPageToken);
    const files = page.files || [];
    state.currentPageToken = page.nextPageToken || null;

    for (const f of files) {
      if (!timeLeft_(ctx)) break;
      if (budget.filesProcessed >= CFG.MAX_FILES_PER_RUN) break;

      budget.filesProcessed++;

      if (isShortcut_(f)) {
        budget.skipped++;
        ctx.logBuffer.push(logObj_("skip_shortcut", { runId: ctx.runId, fileId: f.id, name: f.name }));
        continue;
      }

      if (fileIsUnderBuckets_(ctx, f)) {
        budget.skipped++;
        continue;
      }

      try {
        const bucket = bucketForFile_(f);
        const destFolderId = ctx.bucketIds[bucket] || ctx.bucketIds["OTHER"];

        ensureDestCacheLoaded_(ctx, destFolderId);

        const mv = moveFileWithCollision_(ctx, f, destFolderId);
        if (mv.renamed) budget.collisionsRenamed++;
        if (!ctx.dryRun) budget.moved++;

        ctx.logBuffer.push(logObj_("bucket_move", {
          runId: ctx.runId,
          fileId: f.id,
          fromParents: (f.parents || []).join(","),
          toParent: destFolderId,
          bucket: bucket,
          name: mv.finalName,
          renamed: mv.renamed
        }));
      } catch (e) {
        budget.errors++;
        ctx.logBuffer.push(logObj_("error", { runId: ctx.runId, stage: "bucket_move", fileId: f.id, name: f.name, reason: String(e) }));
      }
    }

    if (!state.currentPageToken) {
      budget.foldersProcessed++;
      try {
        const subs = listSubfolders_(ctx, folderId);
        for (const sid of subs) {
          if (sid === ctx.bucketsRootId) continue;
          if (folderIsUnderBuckets_(ctx, sid)) continue;
          if (folderIsUnderDenylistedRoot_(ctx, sid)) continue;
          state.folderQueue.push(sid);
        }
      } catch (e) {
        budget.errors++;
        ctx.logBuffer.push(logObj_("error", { runId: ctx.runId, stage: "enqueue_subfolders", folderId: folderId, reason: String(e) }));
      }
      state.currentFolderId = null;
      state.currentPageToken = null;
    }

    if (budget.filesProcessed % 200 === 0) saveState_(state);
  }
}

// ========================= DEDUPE (WITHIN BUCKETS) =========================

function dedupeTraversal_(ctx, state, budget) {
  const quarantineId = ctx.bucketIds[CFG.DEDUPE_QUARANTINE_BUCKET];
  ensureDestCacheLoaded_(ctx, quarantineId);

  const index = CFG.ENABLE_PERSISTENT_DEDUPE_INDEX ? loadDedupeIndex_() : null;
  const seen = {}; // per-run map: key -> file

  while (timeLeft_(ctx)) {
    if (budget.dedupeFilesScanned >= CFG.MAX_DEDUPE_FILES_PER_RUN) break;
    if (budget.dedupeFoldersScanned >= CFG.MAX_DEDUPE_FOLDERS_PER_RUN) break;

    if (!state.dedupeCurrentFolderId) {
      if (!state.dedupeQueue || state.dedupeQueue.length === 0) break;
      state.dedupeCurrentFolderId = state.dedupeQueue.shift();
      state.dedupeCurrentPageToken = null;
      saveState_(state);
    }

    const folderId = state.dedupeCurrentFolderId;

    // Enqueue subfolders (PAGINATED FIX)
    try {
      const subs = listSubfolders_(ctx, folderId);
      for (const sid of subs) state.dedupeQueue.push(sid);
    } catch (e) {
      budget.dedupeErrors++;
      ctx.logBuffer.push(logObj_("error", { runId: ctx.runId, stage: "dedupe_enqueue_subfolders", folderId: folderId, reason: String(e) }));
    }

    const page = listFilesInFolder_(ctx, folderId, state.dedupeCurrentPageToken);
    const files = page.files || [];
    state.dedupeCurrentPageToken = page.nextPageToken || null;

    for (const f of files) {
      if (!timeLeft_(ctx)) break;
      if (budget.dedupeFilesScanned >= CFG.MAX_DEDUPE_FILES_PER_RUN) break;

      budget.dedupeFilesScanned++;

      if (isShortcut_(f)) continue;
      if (isUnderQuarantine_(ctx, f)) continue;

      const key = dedupeKey_(f);
      if (!key) continue;
      budget.dedupeCandidates++;

      // PERSISTENT INDEX HIT (FIXED to enforce KEEP_OLDEST across runs)
      if (index && index.map[key]) {
        budget.dedupeIndexHits++;
        const winnerId = index.map[key];

        if (f.id === winnerId) continue;

        try {
          const winnerMeta = getMetaCached_(ctx, winnerId, "id,name,parents,mimeType,modifiedTime,size,md5Checksum");

          // winnerMeta may be null if deleted/missing
          if (!winnerMeta || !winnerMeta.id) {
            index.map[key] = f.id;
            index.order.push(key);
            budget.dedupeIndexAdds++;
            evictDedupeIndexIfNeeded_(index, budget);
            continue;
          }

          const winnerIsOlder = pickOldest_(winnerMeta, f).id === winnerMeta.id;

          if (winnerIsOlder) {
            if (quarantineLoser_(ctx, quarantineId, f, key, winnerId, budget)) budget.dedupeQuarantined++;
          } else {
            // Current file is older: swap winner
            index.map[key] = f.id;
            index.order.push(key);
            budget.dedupeIndexAdds++;
            evictDedupeIndexIfNeeded_(index, budget);

            if (quarantineLoser_(ctx, quarantineId, winnerMeta, key, f.id, budget)) budget.dedupeQuarantined++;
          }
        } catch (e) {
          // If winner lookup fails, re-pin to current file
          index.map[key] = f.id;
          index.order.push(key);
          budget.dedupeIndexAdds++;
          evictDedupeIndexIfNeeded_(index, budget);
        }
        continue;
      }

      // Per-run collision
      if (!seen[key]) {
        seen[key] = f;
        if (index) {
          index.map[key] = f.id;
          index.order.push(key);
          budget.dedupeIndexAdds++;
          evictDedupeIndexIfNeeded_(index, budget);
        }
        continue;
      }

      const winner = pickOldest_(seen[key], f);
      const loser = (winner.id === seen[key].id) ? f : seen[key];
      seen[key] = winner;

      if (index) {
        index.map[key] = winner.id;
        index.order.push(key);
        budget.dedupeIndexAdds++;
        evictDedupeIndexIfNeeded_(index, budget);
      }

      if (quarantineLoser_(ctx, quarantineId, loser, key, winner.id, budget)) budget.dedupeQuarantined++;
    }

    if (!state.dedupeCurrentPageToken) {
      budget.dedupeFoldersScanned++;
      state.dedupeCurrentFolderId = null;
      state.dedupeCurrentPageToken = null;
    }

    if (budget.dedupeFilesScanned % 300 === 0) saveState_(state);
  }

  if (index) saveDedupeIndex_(index);
}

function quarantineLoser_(ctx, quarantineId, loser, key, winnerId, budget) {
  try {
    ensureDestCacheLoaded_(ctx, quarantineId);

    if (!ctx.dryRun) {
      let finalName = loser.name || "";
      if (finalName && destHasNameCached_(ctx, quarantineId, finalName)) {
        finalName = resolveCollisionNameCached_(ctx, quarantineId, finalName);
        callDriveUpdate_(() => Drive.Files.update({ name: finalName }, loser.id));
        budget.dedupeRenamed++;
      }

      const curParents = (loser.parents || []).join(",");
      callDriveUpdate_(() => Drive.Files.update({}, loser.id, { addParents: quarantineId, removeParents: curParents }));

      if (finalName) ctx.destNameCache[quarantineId][finalName] = true;
    }

    ctx.logBuffer.push(logObj_("dedupe_quarantine", {
      runId: ctx.runId,
      key: key,
      loserId: loser.id,
      loserName: loser.name,
      winnerId: winnerId
    }));

    return true;
  } catch (e) {
    budget.dedupeErrors++;
    ctx.logBuffer.push(logObj_("error", { runId: ctx.runId, stage: "dedupe_quarantine", loserId: loser.id, reason: String(e) }));
    return false;
  }
}

// ========================= AGGRESSIVE PRUNE (RECURSIVE, POST-ORDER) =========================

function aggressivePruneEmptyFolders_(ctx, state, budget) {
  // Seed prune roots once: include root itself so empties can be pruned “all the way down” (but never trash root)
  if (!state.pruneRootsSeeded) {
    const roots = [ctx.rootId].concat(seedTopLevelFolders_(ctx)); // excludes BUCKETS+denylisted by name for top-level
    state.pruneStack = [];
    for (const rid of roots) state.pruneStack.push({ id: rid, expanded: false });
    state.pruneRootsSeeded = true;
    saveState_(state);
    ctx.logBuffer.push(logObj_("prune_seed", { runId: ctx.runId, roots: roots.length }));
  }

  let checks = 0;
  let trashed = 0;

  while (timeLeft_(ctx)) {
    if (checks >= CFG.MAX_PRUNE_FOLDER_CHECKS_PER_RUN) break;
    if (trashed >= CFG.MAX_PRUNE_TRASH_PER_RUN) break;
    if (!state.pruneStack || state.pruneStack.length === 0) break;

    const top = state.pruneStack[state.pruneStack.length - 1];

    // Never prune BUCKETS subtree or denylisted subtree; never trash root
    if (top.id === ctx.bucketsRootId || folderIsUnderBuckets_(ctx, top.id) || folderIsUnderDenylistedRoot_(ctx, top.id)) {
      state.pruneStack.pop();
      continue;
    }

    if (!top.expanded) {
      top.expanded = true;
      try {
        const subs = listSubfolders_(ctx, top.id);
        for (const sid of subs) {
          if (sid === ctx.bucketsRootId) continue;
          if (folderIsUnderBuckets_(ctx, sid)) continue;
          if (folderIsUnderDenylistedRoot_(ctx, sid)) continue;
          state.pruneStack.push({ id: sid, expanded: false });
        }
      } catch (e) {
        budget.prunedErrors++;
        ctx.logBuffer.push(logObj_("error", { runId: ctx.runId, stage: "prune_expand", folderId: top.id, reason: String(e) }));
      }
      continue;
    }

    // Post-order: check emptiness and trash if empty (except root)
    state.pruneStack.pop();

    try {
      checks++;
      budget.pruneFolderChecks++;

      if (top.id === ctx.rootId) continue;

      const empty = folderIsEmpty_(ctx, top.id);
      if (!empty) continue;

      if (!ctx.dryRun) {
        callDriveUpdate_(() => Drive.Files.update({ trashed: true }, top.id));
        trashed++;
        budget.prunedFoldersTrashed++;
      }

      ctx.logBuffer.push(logObj_("prune_empty_folder", {
        runId: ctx.runId,
        folderId: top.id,
        op: ctx.dryRun ? "plan" : "trash"
      }));
    } catch (e) {
      budget.prunedErrors++;
      ctx.logBuffer.push(logObj_("error", { runId: ctx.runId, stage: "prune_check_or_trash", folderId: top.id, reason: String(e) }));
    }

    if ((checks % 200) === 0) saveState_(state);
  }

  saveState_(state);
}

// ========================= MOVE + COLLISION (CACHED) =========================

function moveFileWithCollision_(ctx, fileObj, destFolderId) {
  const fileId = fileObj.id;
  const originalName = fileObj.name || "";

  let finalName = originalName;
  let renamed = false;

  if (finalName && destHasNameCached_(ctx, destFolderId, finalName)) {
    finalName = resolveCollisionNameCached_(ctx, destFolderId, finalName);
    renamed = (finalName !== originalName);
    if (!ctx.dryRun) callDriveUpdate_(() => Drive.Files.update({ name: finalName }, fileId));
  }

  if (!ctx.dryRun) {
    const curParents = (fileObj.parents || []).join(",");
    callDriveUpdate_(() => Drive.Files.update({}, fileId, { addParents: destFolderId, removeParents: curParents }));
    if (finalName) ctx.destNameCache[destFolderId][finalName] = true;
  }

  return { renamed: renamed, finalName: finalName };
}

function ensureDestCacheLoaded_(ctx, folderId) {
  if (ctx.destNameCache[folderId]) return;

  const names = {};
  let pageToken = null;
  let loaded = 0;
  const LOAD_LIMIT = 5000;

  do {
    if (!timeLeft_(ctx)) break;

    const resp = callDriveList_(() => Drive.Files.list({
      q: `'${folderId}' in parents and trashed = false`,
      fields: "nextPageToken, files(name)",
      pageSize: 200,
      pageToken: pageToken || undefined,
      supportsAllDrives: false
    }));

    const files = resp.files || [];
    for (const f of files) {
      if (!f.name) continue;
      names[f.name] = true;
      loaded++;
      if (loaded >= LOAD_LIMIT) break;
    }

    pageToken = resp.nextPageToken || null;
    if (loaded >= LOAD_LIMIT) break;
  } while (pageToken);

  ctx.destNameCache[folderId] = names;
}

function destHasNameCached_(ctx, folderId, name) {
  const cache = ctx.destNameCache[folderId];
  if (!cache) return false;
  return !!cache[name];
}

function resolveCollisionNameCached_(ctx, folderId, name) {
  const idx = name.lastIndexOf(".");
  const base = idx >= 0 ? name.substring(0, idx) : name;
  const ext = idx >= 0 ? name.substring(idx) : "";

  if (CFG.COLLISION_POLICY === "timestamp") {
    const ts = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), "yyyyMMdd_HHmmss");
    const cand = `${base}__${ts}${ext}`;
    if (!destHasNameCached_(ctx, folderId, cand)) return cand;
  }

  for (let i = 1; i <= 9999; i++) {
    const cand = `${base} (${i})${ext}`;
    if (!destHasNameCached_(ctx, folderId, cand)) return cand;
  }

  const ts2 = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), "yyyyMMdd_HHmmss");
  return `${base}__COLLISION__${ts2}${ext}`;
}

// ========================= CLASSIFICATION =========================

function bucketForFile_(fileObj) {
  const mt = (fileObj.mimeType || "").toLowerCase();
  if (CFG.MIME_BUCKETS[mt]) return CFG.MIME_BUCKETS[mt];

  const ext = extOf_(fileObj.name || "");
  for (const [bucket, exts] of Object.entries(CFG.BUCKETS)) {
    if (exts.indexOf(ext) >= 0) return bucket;
  }
  return "OTHER";
}

function extOf_(name) {
  const idx = name.lastIndexOf(".");
  if (idx < 0) return "";
  return name.substring(idx).toLowerCase();
}

// ========================= DEDUPE KEYING =========================

function dedupeKey_(fileObj) {
  const md5 = fileObj.md5Checksum || "";
  if (md5) return "md5:" + md5;

  const name = fileObj.name || "";
  const size = String(fileObj.size || fileObj.fileSize || "0");
  const mt = fileObj.mimeType || "";
  if (!name && !mt) return null;
  return "nss:" + [name, size, mt].join("|");
}

function pickOldest_(a, b) {
  const am = a.modifiedTime || "";
  const bm = b.modifiedTime || "";
  if (am && bm) return (am <= bm) ? a : b;
  return (String(a.id) <= String(b.id)) ? a : b;
}

// ========================= CONTAINMENT: BUCKETS + DENYLIST SUBTREES =========================

function fileIsUnderBuckets_(ctx, fileObj) {
  const parents = fileObj.parents || [];
  for (const pid of parents) {
    if (pid === ctx.bucketsRootId) return true;
    if (folderIsUnderBuckets_(ctx, pid)) return true;
  }
  return false;
}

function folderIsUnderBuckets_(ctx, folderId) {
  if (!folderId) return false;
  if (folderId === ctx.bucketsRootId) return true;
  if (ctx.folderUnderBucketsCache.hasOwnProperty(folderId)) return ctx.folderUnderBucketsCache[folderId];

  const visited = {};
  let cur = folderId;

  while (cur) {
    if (cur === ctx.bucketsRootId) {
      for (const k in visited) ctx.folderUnderBucketsCache[k] = true;
      ctx.folderUnderBucketsCache[folderId] = true;
      return true;
    }
    if (ctx.folderUnderBucketsCache.hasOwnProperty(cur)) {
      const v = ctx.folderUnderBucketsCache[cur];
      for (const k in visited) ctx.folderUnderBucketsCache[k] = v;
      ctx.folderUnderBucketsCache[folderId] = v;
      return v;
    }
    if (visited[cur]) break;
    visited[cur] = true;

    const parent = getParentOfFolderCached_(ctx, cur);
    cur = parent;
  }

  for (const k in visited) ctx.folderUnderBucketsCache[k] = false;
  ctx.folderUnderBucketsCache[folderId] = false;
  return false;
}

function folderIsUnderDenylistedRoot_(ctx, folderId) {
  if (!folderId) return false;
  if (ctx.folderUnderDenyRootCache.hasOwnProperty(folderId)) return ctx.folderUnderDenyRootCache[folderId];

  const denyTopIds = getDenylistedTopLevelIds_(ctx);

  // Fast path: if folder is top-level (parent == root), check deny list directly
  const parent = getParentOfFolderCached_(ctx, folderId);
  if (parent === ctx.rootId) {
    const v = !!denyTopIds[folderId];
    ctx.folderUnderDenyRootCache[folderId] = v;
    return v;
  }

  const visited = {};
  let cur = folderId;

  while (cur) {
    if (denyTopIds[cur]) {
      for (const k in visited) ctx.folderUnderDenyRootCache[k] = true;
      ctx.folderUnderDenyRootCache[folderId] = true;
      return true;
    }
    if (ctx.folderUnderDenyRootCache.hasOwnProperty(cur)) {
      const v = ctx.folderUnderDenyRootCache[cur];
      for (const k in visited) ctx.folderUnderDenyRootCache[k] = v;
      ctx.folderUnderDenyRootCache[folderId] = v;
      return v;
    }
    if (visited[cur]) break;
    visited[cur] = true;

    cur = getParentOfFolderCached_(ctx, cur);
  }

  for (const k in visited) ctx.folderUnderDenyRootCache[k] = false;
  ctx.folderUnderDenyRootCache[folderId] = false;
  return false;
}

function getDenylistedTopLevelIds_(ctx) {
  if (ctx._denyTopIds) return ctx._denyTopIds;

  const root = DriveApp.getRootFolder();
  const it = root.getFolders();
  const out = {};
  while (it.hasNext()) {
    const f = it.next();
    const name = f.getName();
    if (ctx.denyNameSet[name]) out[f.getId()] = true;
  }
  ctx._denyTopIds = out;
  return out;
}

function isUnderQuarantine_(ctx, fileObj) {
  const qid = ctx.bucketIds[CFG.DEDUPE_QUARANTINE_BUCKET];
  const parents = fileObj.parents || [];
  for (const pid of parents) if (pid === qid) return true;
  return false;
}

function isShortcut_(fileObj) {
  return (fileObj.mimeType || "").toLowerCase() === "application/vnd.google-apps.shortcut";
}

// ========================= DRIVE LISTING HELPERS (PAGINATED FIX) =========================

function listFilesInFolder_(ctx, folderId, pageToken) {
  const q = [
    `'${folderId}' in parents`,
    "trashed = false",
    "mimeType != 'application/vnd.google-apps.folder'"
  ].join(" and ");

  const args = {
    q: q,
    fields: "nextPageToken, files(id,name,parents,mimeType,modifiedTime,size,md5Checksum)",
    pageSize: 200,
    supportsAllDrives: false
  };
  if (pageToken) args.pageToken = pageToken;

  return callDriveList_(() => Drive.Files.list(args));
}

function listSubfolders_(ctx, folderId) {
  const q = [
    `'${folderId}' in parents`,
    "trashed = false",
    "mimeType = 'application/vnd.google-apps.folder'"
  ].join(" and ");

  const out = [];
  let pageToken = null;

  do {
    if (!timeLeft_(ctx)) break;

    const resp = callDriveList_(() => Drive.Files.list({
      q: q,
      fields: "nextPageToken, files(id,name,parents,mimeType)",
      pageSize: 200,
      pageToken: pageToken || undefined,
      supportsAllDrives: false
    }));

    const files = resp.files || [];
    for (const f of files) out.push(f.id);

    pageToken = resp.nextPageToken || null;
  } while (pageToken && timeLeft_(ctx));

  return out;
}

function folderIsEmpty_(ctx, folderId) {
  const resp = callDriveList_(() => Drive.Files.list({
    q: `'${folderId}' in parents and trashed = false`,
    fields: "files(id)",
    pageSize: 1,
    supportsAllDrives: false
  }));
  return !(resp.files && resp.files.length > 0);
}

// ========================= TOP-LEVEL SEEDING =========================

function seedTopLevelFolders_(ctx) {
  const root = DriveApp.getRootFolder();
  const out = [];
  const it = root.getFolders();
  while (it.hasNext()) {
    const f = it.next();
    const name = f.getName();
    if (name === CFG.BUCKETS_DIR_NAME) continue;
    if (ctx.denyNameSet[name]) continue;
    out.push(f.getId());
  }
  return out;
}

// ========================= FOLDER CREATION (ROBUST NAME MATCH) =========================

function ensureFolderUnderParent_(parentFolderId, name) {
  const q = [
    `'${parentFolderId}' in parents`,
    `mimeType = 'application/vnd.google-apps.folder'`,
    "trashed = false",
    `name = '${escapeDriveQ_(name)}'`
  ].join(" and ");

  const resp = callDriveList_(() => Drive.Files.list({
    q: q,
    fields: "files(id,name)",
    pageSize: 10,
    supportsAllDrives: false
  }));

  if (resp.files && resp.files.length > 0) {
    // If multiple, choose first; this is safe; log best-effort
    if (resp.files.length > 1) {
      Logger.log("WARN multiple folders with same name under parent; using first. name=" + name + " count=" + resp.files.length);
    }
    return resp.files[0].id;
  }

  const created = callDriveInsert_(() => Drive.Files.insert({
    name: name,
    mimeType: "application/vnd.google-apps.folder",
    parents: [{ id: parentFolderId }]
  }));

  return created.id;
}

function escapeDriveQ_(s) {
  // Escape single quotes for Drive query literal
  return String(s).replace(/'/g, "\\'");
}

// ========================= LOGGING =========================

function ensureLogSheet_() {
  const props = PropertiesService.getScriptProperties();
  const existingId = props.getProperty("BUCKETIZER_LOG_SHEET_ID");
  if (existingId) return { spreadsheetId: existingId, tabName: CFG.LOG_SHEET_TAB };

  const ss = SpreadsheetApp.create(CFG.LOG_SPREADSHEET_NAME);
  const sh = ss.getSheets()[0];
  sh.setName(CFG.LOG_SHEET_TAB);
  sh.appendRow(["ts", "event", "dryRun", "runId", "payload_json"]);

  const id = ss.getId();
  props.setProperty("BUCKETIZER_LOG_SHEET_ID", id);
  return { spreadsheetId: id, tabName: CFG.LOG_SHEET_TAB };
}

function logObj_(event, payload) {
  return { ts: new Date().toISOString(), event: event, payload: payload || {} };
}

function flushLogs_(ctx, budget) {
  const ss = SpreadsheetApp.openById(ctx.logSpreadsheetId);
  const sh = ss.getSheetByName(ctx.logTabName) || ss.getSheets()[0];

  const currentRows = sh.getLastRow();
  if (currentRows >= CFG.LOG_MAX_ROWS_SOFT) {
    Logger.log("LOG_SOFT_LIMIT_REACHED rows=" + currentRows + " :: summary=" + JSON.stringify(budget));
    try {
      sh.appendRow([new Date().toISOString(), "run_summary_log_rollover", String(ctx.dryRun), ctx.runId, JSON.stringify(budget)]);
    } catch (e) {
      Logger.log("LOG_APPEND_FAIL " + String(e));
    }
    return;
  }

  const rows = [];
  for (const item of ctx.logBuffer) rows.push([item.ts, item.event, String(ctx.dryRun), ctx.runId, JSON.stringify(item.payload)]);
  rows.push([new Date().toISOString(), "run_summary", String(ctx.dryRun), ctx.runId, JSON.stringify(budget)]);

  if (rows.length === 0) return;
  sh.getRange(sh.getLastRow() + 1, 1, rows.length, 5).setValues(rows);
}

function finalizeRun_(ctx, budget) {
  try { flushLogs_(ctx, budget); } catch (e) { Logger.log("FLUSH_LOGS_FAIL " + String(e)); }
  Logger.log("SUMMARY " + (ctx.dryRun ? "[DRY]" : "[COMMIT]") + " " + JSON.stringify(budget));
}

// ========================= DASHBOARD =========================

function ensureDashboardSheet_(spreadsheetId) {
  const ss = SpreadsheetApp.openById(spreadsheetId);
  let sh = ss.getSheetByName(DASH.TAB_NAME);
  if (!sh) sh = ss.insertSheet(DASH.TAB_NAME);

  if (sh.getLastRow() === 0) {
    sh.getRange(1, 1, 1, 2).setValues([["BUCKETIZER STATUS", "VALUE"]]);
    sh.getRange(1, 1, 1, 2).setFontWeight("bold");
    sh.setFrozenRows(1);
  }
  return sh;
}

function dashboardSnapshot_(ctx, state, budget) {
  const now = new Date();
  const elapsedMs = Date.now() - ctx.startedMs;

  const folderQueueLen = (state.folderQueue && state.folderQueue.length) ? state.folderQueue.length : 0;
  const dedupeQueueLen = (state.dedupeQueue && state.dedupeQueue.length) ? state.dedupeQueue.length : 0;
  const pruneStackLen  = (state.pruneStack  && state.pruneStack.length)  ? state.pruneStack.length  : 0;

  const currentFolder = state.currentFolderId || "";
  const currentFolderPaging = state.currentPageToken ? "PAGING" : "";

  const dedupeCurrentFolder = state.dedupeCurrentFolderId || "";
  const dedupePaging = state.dedupeCurrentPageToken ? "PAGING" : "";

  const pruneSeeded = (typeof state.pruneRootsSeeded === "boolean") ? String(state.pruneRootsSeeded) : "false";

  const likelyDone =
    folderQueueLen === 0 && !state.currentFolderId &&
    dedupeQueueLen === 0 && !state.dedupeCurrentFolderId &&
    pruneStackLen === 0;

  return [
    ["Last update (local)", Utilities.formatDate(now, Session.getScriptTimeZone(), "yyyy-MM-dd HH:mm:ss")],
    ["Run ID", ctx.runId],
    ["Dry run", String(ctx.dryRun)],
    ["Elapsed (ms)", String(elapsedMs)],
    ["Elapsed (s)", String(Math.round(elapsedMs / 1000))],

    ["Bucketize: files processed (run)", String(budget.filesProcessed || 0)],
    ["Bucketize: moved (run)", String(budget.moved || 0)],
    ["Bucketize: folders processed (run)", String(budget.foldersProcessed || 0)],
    ["Bucketize: errors (run)", String(budget.errors || 0)],
    ["Bucketize: queue remaining", String(folderQueueLen)],
    ["Bucketize: current folder", currentFolder],
    ["Bucketize: current folder status", currentFolderPaging],

    ["Dedupe: files scanned (run)", String(budget.dedupeFilesScanned || 0)],
    ["Dedupe: quarantined (run)", String(budget.dedupeQuarantined || 0)],
    ["Dedupe: renamed (run)", String(budget.dedupeRenamed || 0)],
    ["Dedupe: errors (run)", String(budget.dedupeErrors || 0)],
    ["Dedupe: index hits (run)", String(budget.dedupeIndexHits || 0)],
    ["Dedupe: queue remaining", String(dedupeQueueLen)],
    ["Dedupe: current folder", dedupeCurrentFolder],
    ["Dedupe: current folder status", dedupePaging],

    ["Prune: folder checks (run)", String(budget.pruneFolderChecks || 0)],
    ["Prune: trashed empty folders (run)", String(budget.prunedFoldersTrashed || 0)],
    ["Prune: errors (run)", String(budget.prunedErrors || 0)],
    ["Prune: seeded", pruneSeeded],
    ["Prune: stack remaining", String(pruneStackLen)],

    ["Likely complete", likelyDone ? "YES" : "NO"]
  ];
}

function updateDashboard_(ctx, state, budget) {
  const sh = ensureDashboardSheet_(ctx.logSpreadsheetId);
  const kv = dashboardSnapshot_(ctx, state, budget);

  const maxRows = Math.max(sh.getLastRow(), DASH.KV_START_ROW + kv.length + 10);
  sh.getRange(DASH.KV_START_ROW, 1, maxRows - DASH.KV_START_ROW + 1, 2).clearContent();

  sh.getRange(DASH.KV_START_ROW, 1, kv.length, 2).setValues(kv);
  sh.getRange(DASH.KV_START_ROW, 1, kv.length, 1).setFontWeight("bold");

  sh.autoResizeColumns(1, 2);
}

// ========================= STATE I/O =========================

function loadState_() {
  const props = PropertiesService.getScriptProperties();
  const raw = props.getProperty("BUCKETIZER_STATE_V4_2");
  if (!raw) return { initialized: false, stateVersion: "4.2", pruneStack: [], pruneRootsSeeded: false };
  try {
    const s = JSON.parse(raw);
    if (!s.stateVersion) s.stateVersion = "4.2";
    if (!s.pruneStack) s.pruneStack = [];
    if (typeof s.pruneRootsSeeded !== "boolean") s.pruneRootsSeeded = false;
    return s;
  } catch (e) {
    return { initialized: false, stateVersion: "4.2", pruneStack: [], pruneRootsSeeded: false };
  }
}

function saveState_(state) {
  PropertiesService.getScriptProperties().setProperty("BUCKETIZER_STATE_V4_2", JSON.stringify(state));
}

// ========================= PERSISTENT DEDUPE INDEX =========================

function loadDedupeIndex_() {
  const props = PropertiesService.getScriptProperties();
  const raw = props.getProperty(CFG.DEDUPE_INDEX_PROP_KEY);
  if (!raw) return { map: {}, order: [] };
  try {
    const obj = JSON.parse(raw);
    if (!obj.map) obj.map = {};
    if (!obj.order) obj.order = [];
    return obj;
  } catch (e) {
    return { map: {}, order: [] };
  }
}

function saveDedupeIndex_(idx) {
  idx.order = compactOrder_(idx);
  PropertiesService.getScriptProperties().setProperty(CFG.DEDUPE_INDEX_PROP_KEY, JSON.stringify(idx));
}

function evictDedupeIndexIfNeeded_(idx, budget) {
  if (Object.keys(idx.map).length <= CFG.DEDUPE_INDEX_MAX_KEYS) return;

  idx.order = compactOrder_(idx);

  const over = Object.keys(idx.map).length - CFG.DEDUPE_INDEX_MAX_KEYS;
  let evicted = 0;
  while (evicted < over && idx.order.length > 0) {
    const k = idx.order.shift();
    if (idx.map.hasOwnProperty(k)) {
      delete idx.map[k];
      evicted++;
    }
  }
  budget.dedupeIndexEvicts += evicted;
}

function compactOrder_(idx) {
  const seen = {};
  const out = [];
  for (let i = idx.order.length - 1; i >= 0; i--) {
    const k = idx.order[i];
    if (seen[k]) continue;
    seen[k] = true;
    out.push(k);
  }
  out.reverse();

  const filtered = [];
  for (const k of out) if (idx.map.hasOwnProperty(k)) filtered.push(k);
  return filtered;
}

// ========================= META + PARENT CACHE HELPERS =========================

function getMetaCached_(ctx, id, fields) {
  if (!id) return null;
  if (ctx.metaCache[id]) return ctx.metaCache[id];

  const meta = callDriveGet_(() => Drive.Files.get(id, { fields: fields, supportsAllDrives: false }));
  // Best-effort cache bound
  ctx.metaCache[id] = meta || null;
  enforceCacheBounds_(ctx);
  return ctx.metaCache[id];
}

function getParentOfFolderCached_(ctx, folderId) {
  if (!folderId) return null;
  if (ctx.parentOfFolderCache.hasOwnProperty(folderId)) return ctx.parentOfFolderCache[folderId];

  const meta = getMetaCached_(ctx, folderId, "id,parents,mimeType");
  if (!meta || meta.mimeType !== "application/vnd.google-apps.folder") {
    ctx.parentOfFolderCache[folderId] = null;
    return null;
  }
  const ps = meta.parents || [];
  const parent = ps.length > 0 ? ps[0] : null;
  ctx.parentOfFolderCache[folderId] = parent;
  enforceCacheBounds_(ctx);
  return parent;
}

function enforceCacheBounds_(ctx) {
  // best-effort: if caches balloon, drop them entirely (safe; performance-only)
  if (Object.keys(ctx.parentOfFolderCache).length > CFG.MAX_PARENT_CACHE_ENTRIES) ctx.parentOfFolderCache = {};
  if (Object.keys(ctx.metaCache).length > CFG.MAX_META_CACHE_ENTRIES) ctx.metaCache = {};
}

// ========================= DRIVE API BACKOFF WRAPPER =========================

function callDriveList_(fn) { return callWithBackoff_(fn, "Drive.Files.list"); }
function callDriveGet_(fn) { return callWithBackoff_(fn, "Drive.Files.get"); }
function callDriveUpdate_(fn) { return callWithBackoff_(fn, "Drive.Files.update"); }
function callDriveInsert_(fn) { return callWithBackoff_(fn, "Drive.Files.insert"); }

function callWithBackoff_(fn, label) {
  let attempt = 0;
  while (true) {
    try {
      return fn();
    } catch (e) {
      attempt++;
      const msg = String(e);
      const retryable =
        msg.indexOf("Rate Limit") >= 0 ||
        msg.indexOf("quota") >= 0 ||
        msg.indexOf("429") >= 0 ||
        msg.indexOf("503") >= 0 ||
        msg.indexOf("500") >= 0 ||
        msg.indexOf("Service invoked too many times") >= 0;

      if (!retryable || attempt >= CFG.RETRY_MAX_ATTEMPTS) {
        throw new Error(label + " failed after " + attempt + " attempts: " + msg);
      }

      const base = CFG.RETRY_BASE_SLEEP_MS * Math.pow(2, attempt - 1);
      const jitter = Math.floor(Math.random() * CFG.RETRY_JITTER_MS);
      Utilities.sleep(base + jitter);
    }
  }
}

// ========================= UTILS =========================

function toSet_(arr) {
  const s = {};
  for (const x of (arr || [])) s[x] = true;
  return s;
}

function timeLeft_(ctx) {
  return (Date.now() - ctx.startedMs) <= CFG.SOFT_TIME_LIMIT_MS;
}
) {
  
}
