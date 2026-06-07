-- CRDT Relational Sync Engine — Shadow Tables
-- These tables store the CRDT metadata that enables offline-first sync
-- with relational safety guarantees.

-- Schema registry: tracks which tables are CRDT-managed
CREATE TABLE IF NOT EXISTS _crdt_schema (
    table_name      TEXT PRIMARY KEY,
    primary_key_col TEXT NOT NULL DEFAULT 'id',
    columns_json    TEXT NOT NULL,       -- JSON list of column definitions
    foreign_keys_json TEXT DEFAULT '[]', -- JSON list of FK definitions
    unique_cols_json TEXT DEFAULT '[]',  -- JSON list of unique column names
    on_tombstone_policy TEXT NOT NULL DEFAULT 'preserve',  -- cascade|nullify|preserve
    registered_at   TEXT NOT NULL        -- HLC timestamp when registered
);

-- Shadow table: one row per (table, row_id, column, writer) cell version
CREATE TABLE IF NOT EXISTS _crdt_cells (
    table_name  TEXT NOT NULL,
    row_id      TEXT NOT NULL,
    col_name    TEXT NOT NULL,
    writer_id   TEXT NOT NULL,
    value       TEXT,
    hlc_ts      TEXT NOT NULL,
    is_winner   INTEGER DEFAULT 1,
    PRIMARY KEY (table_name, row_id, col_name, writer_id)
);

CREATE INDEX IF NOT EXISTS idx_crdt_cells_lookup
    ON _crdt_cells(table_name, row_id, col_name, is_winner);

CREATE INDEX IF NOT EXISTS idx_crdt_cells_sync
    ON _crdt_cells(hlc_ts);

-- Tombstones: soft-deleted rows held alive for FK safety
CREATE TABLE IF NOT EXISTS _tombstones (
    table_name      TEXT NOT NULL,
    row_id          TEXT NOT NULL,
    deleted_at_hlc  TEXT NOT NULL,
    deleted_by      TEXT NOT NULL,
    ref_count       INTEGER DEFAULT 0,
    is_resolved     INTEGER DEFAULT 0,
    PRIMARY KEY (table_name, row_id)
);

-- Conflict artifacts: losing rows from uniqueness conflicts
CREATE TABLE IF NOT EXISTS _conflict_artifacts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    original_table  TEXT NOT NULL,
    winning_row_id  TEXT,
    losing_row_json TEXT NOT NULL,
    conflict_type   TEXT NOT NULL,
    conflicting_key TEXT NOT NULL,
    winner_writer   TEXT NOT NULL,
    loser_writer    TEXT NOT NULL,
    detected_at     TEXT NOT NULL,
    resolved        INTEGER DEFAULT 0
);

-- Vector clocks: per-peer tracking for compaction
CREATE TABLE IF NOT EXISTS _vector_clocks (
    peer_id       TEXT NOT NULL,
    writer_id     TEXT NOT NULL,
    max_hlc_ts    TEXT NOT NULL,
    PRIMARY KEY (peer_id, writer_id)
);

-- Row existence tracking: which rows are alive vs tombstoned
CREATE TABLE IF NOT EXISTS _crdt_row_state (
    table_name  TEXT NOT NULL,
    row_id      TEXT NOT NULL,
    is_deleted  INTEGER DEFAULT 0,
    created_hlc TEXT NOT NULL,
    writer_id   TEXT NOT NULL,
    PRIMARY KEY (table_name, row_id)
);
