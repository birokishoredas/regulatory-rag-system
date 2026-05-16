-- UUID support
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- --------------------------------------------------
-- Evaluation Logs Table (UPDATED)
-- --------------------------------------------------

CREATE TABLE evaluation_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Core fields
    query TEXT,
    answer TEXT,                     
    status TEXT,

    -- Performance
    latency_ms INT,

    -- Core metrics
    grounding_score FLOAT,
    judge_score FLOAT,
    citation_score FLOAT,

    -- Optional RAGAS metrics
    ragas JSONB,

    -- Full payload (debugging / audit)
    data JSONB,

    -- Timestamp
    created_at TIMESTAMP DEFAULT NOW()
);

-- --------------------------------------------------
-- Indexes
-- --------------------------------------------------

-- 1. Time-based (most important)
CREATE INDEX idx_eval_created_at
ON evaluation_logs (created_at DESC);

-- 2. Query search (optional but useful)
CREATE INDEX idx_eval_query
ON evaluation_logs USING GIN (to_tsvector('english', query));

-- 3. JSONB flexible querying
CREATE INDEX idx_eval_data_gin
ON evaluation_logs USING GIN (data);

-- 4. RAGAS metrics (optional analytics)
CREATE INDEX idx_eval_ragas_gin
ON evaluation_logs USING GIN (ragas);