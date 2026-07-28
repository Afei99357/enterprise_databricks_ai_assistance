-- Bootstrap: Create conversation memory tables for the WNV assistant.
-- Run this once via Databricks SQL or a notebook before deploying the agent.
-- Tables are created idempotently (CREATE IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS eliao.wnv_demo.assistant_turns (
    user_id STRING NOT NULL,
    conversation_id STRING NOT NULL,
    turn_id STRING NOT NULL,
    turn_number BIGINT NOT NULL,
    role STRING NOT NULL,
    content STRING NOT NULL,

    route STRING,
    generated_sql STRING,
    result_summary STRING,
    analytics_state_json STRING,
    citations_json STRING,

    created_at TIMESTAMP NOT NULL
)
USING DELTA;

-- Optional: create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_turns_user_conversation
ON eliao.wnv_demo.assistant_turns (user_id, conversation_id);
