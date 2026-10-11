-- ---------------------------------------------------------------------------
-- 001_initial_schema.sql — LinkedIn Job Agent core schema
--
-- Every statement is idempotent (IF NOT EXISTS / ON CONFLICT) so that a fresh
-- volume init and a re-run against an existing database both succeed. The CI
-- "migration cleanliness" gate applies this file to a brand-new database with
-- ON_ERROR_STOP=1, so any non-idempotent statement fails the build.
--
-- Design notes
--   * Status columns are TEXT + CHECK rather than native ENUMs. Adding a value
--     to a CHECK is a one-line ALTER; adding one to an ENUM in a transaction is
--     restricted in older PG and is a common migration foot-gun.
--   * TIMESTAMPTZ everywhere. The host may run any timezone; storing naive local
--     time would make the daily rate-limit window ambiguous across DST.
--   * The unique index on applications(job_id) is the structural guarantee
--     behind "never apply twice" — it holds even if application code is buggy.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Resumes ────────────────────────────────────────────────────────────────
-- One row per CV variant. `content_text` is the extracted plain text and is the
-- ONLY thing ever sent to the AI layer: it is the ground truth that bounds what
-- a cover letter is allowed to claim.
CREATE TABLE IF NOT EXISTS resumes (
    id                  BIGSERIAL PRIMARY KEY,
    label               TEXT        NOT NULL UNIQUE,
    file_path           TEXT        NOT NULL,
    content_text        TEXT        NOT NULL,
    skills              TEXT[]      NOT NULL DEFAULT '{}',
    years_experience    NUMERIC(4,1),
    target_roles        TEXT[]      NOT NULL DEFAULT '{}',
    is_default          BOOLEAN     NOT NULL DEFAULT FALSE,
    is_active           BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- At most one default resume. A partial unique index enforces it in the
-- database so a race between two API calls cannot produce two defaults.
CREATE UNIQUE INDEX IF NOT EXISTS ux_resumes_single_default
    ON resumes (is_default) WHERE is_default;

-- ── Search profiles ────────────────────────────────────────────────────────
-- A saved LinkedIn search plus the policy applied to its results.
CREATE TABLE IF NOT EXISTS search_profiles (
    id                          BIGSERIAL PRIMARY KEY,
    name                        TEXT        NOT NULL UNIQUE,
    keywords                    TEXT        NOT NULL,
    location                    TEXT        NOT NULL DEFAULT '',
    remote_only                 BOOLEAN     NOT NULL DEFAULT FALSE,
    easy_apply_only             BOOLEAN     NOT NULL DEFAULT TRUE,
    experience_levels           TEXT[]      NOT NULL DEFAULT '{}',
    job_types                   TEXT[]      NOT NULL DEFAULT '{}',
    date_posted                 TEXT        NOT NULL DEFAULT 'past_week'
        CHECK (date_posted IN ('past_24h','past_week','past_month','any')),
    exclude_companies           TEXT[]      NOT NULL DEFAULT '{}',
    exclude_title_keywords      TEXT[]      NOT NULL DEFAULT '{}',
    min_score                   INTEGER     CHECK (min_score BETWEEN 0 AND 100),
    max_applications_per_run    INTEGER     CHECK (max_applications_per_run BETWEEN 1 AND 20),
    enabled                     BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Runs ───────────────────────────────────────────────────────────────────
-- One execution of "search → score → apply". Pause/resume is modelled as a
-- status transition so a run can survive a worker restart.
CREATE TABLE IF NOT EXISTS runs (
    id                  BIGSERIAL PRIMARY KEY,
    search_profile_id   BIGINT      REFERENCES search_profiles(id) ON DELETE SET NULL,
    status              TEXT        NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','running','paused','completed','failed','cancelled')),
    trigger             TEXT        NOT NULL DEFAULT 'manual'
        CHECK (trigger IN ('manual','schedule','n8n','api')),
    pause_reason        TEXT,
    jobs_scanned        INTEGER     NOT NULL DEFAULT 0,
    jobs_scored         INTEGER     NOT NULL DEFAULT 0,
    applications_created INTEGER    NOT NULL DEFAULT 0,
    applications_submitted INTEGER  NOT NULL DEFAULT 0,
    consecutive_failures INTEGER    NOT NULL DEFAULT 0,
    error               TEXT,
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_runs_status     ON runs (status);
CREATE INDEX IF NOT EXISTS ix_runs_created_at ON runs (created_at DESC);

-- ── Jobs ───────────────────────────────────────────────────────────────────
-- Deduplicated catalogue of everything ever seen. `linkedin_job_id` is the
-- natural key; UNIQUE on it means re-scanning a search is free and idempotent.
CREATE TABLE IF NOT EXISTS jobs (
    id                  BIGSERIAL PRIMARY KEY,
    linkedin_job_id     TEXT        NOT NULL UNIQUE,
    url                 TEXT        NOT NULL,
    title               TEXT        NOT NULL,
    company             TEXT        NOT NULL DEFAULT '',
    location            TEXT        NOT NULL DEFAULT '',
    workplace_type      TEXT,
    employment_type     TEXT,
    seniority           TEXT,
    description         TEXT        NOT NULL DEFAULT '',
    easy_apply          BOOLEAN     NOT NULL DEFAULT FALSE,
    applicant_count     INTEGER,
    posted_at           TIMESTAMPTZ,
    discovered_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    first_seen_run_id   BIGINT      REFERENCES runs(id) ON DELETE SET NULL,
    raw                 JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_jobs_company       ON jobs (company);
CREATE INDEX IF NOT EXISTS ix_jobs_discovered_at ON jobs (discovered_at DESC);
CREATE INDEX IF NOT EXISTS ix_jobs_easy_apply    ON jobs (easy_apply) WHERE easy_apply;

-- ── Job scores ─────────────────────────────────────────────────────────────
-- Compatibility of one job against one resume. Kept as its own table (rather
-- than columns on `jobs`) because every resume gets its own score, and picking
-- "the best resume" is then a plain ORDER BY.
CREATE TABLE IF NOT EXISTS job_scores (
    id                  BIGSERIAL PRIMARY KEY,
    job_id              BIGINT      NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    resume_id           BIGINT      NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    score               INTEGER     NOT NULL CHECK (score BETWEEN 0 AND 100),
    verdict             TEXT        NOT NULL
        CHECK (verdict IN ('strong_match','possible_match','weak_match','not_a_match')),
    reasoning           TEXT        NOT NULL DEFAULT '',
    matched_skills      TEXT[]      NOT NULL DEFAULT '{}',
    missing_skills      TEXT[]      NOT NULL DEFAULT '{}',
    concerns            TEXT[]      NOT NULL DEFAULT '{}',
    provider            TEXT        NOT NULL DEFAULT '',
    model               TEXT        NOT NULL DEFAULT '',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (job_id, resume_id)
);

CREATE INDEX IF NOT EXISTS ix_job_scores_score ON job_scores (score DESC);

-- ── Applications ───────────────────────────────────────────────────────────
-- The unique index on job_id is the "never apply twice" guarantee.
CREATE TABLE IF NOT EXISTS applications (
    id                  BIGSERIAL PRIMARY KEY,
    job_id              BIGINT      NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    run_id              BIGINT      REFERENCES runs(id) ON DELETE SET NULL,
    resume_id           BIGINT      REFERENCES resumes(id) ON DELETE SET NULL,
    score               INTEGER     CHECK (score BETWEEN 0 AND 100),
    status              TEXT        NOT NULL DEFAULT 'pending_review'
        CHECK (status IN (
            'pending_review',   -- waiting for a human to approve
            'approved',         -- human said yes; queued for submission
            'submitting',       -- worker has picked it up
            'submitted',        -- successfully sent to LinkedIn
            'skipped',          -- policy or human declined
            'needs_manual',     -- blocked: CAPTCHA, unknown question, low confidence
            'failed'            -- automation error
        )),
    cover_letter        TEXT,
    cover_letter_flagged BOOLEAN    NOT NULL DEFAULT FALSE,
    cover_letter_flags  TEXT[]      NOT NULL DEFAULT '{}',
    review_reason       TEXT,
    skip_reason         TEXT,
    notes               TEXT,
    error               TEXT,
    screenshot_path     TEXT,
    attempts            INTEGER     NOT NULL DEFAULT 0,
    reviewed_at         TIMESTAMPTZ,
    submitted_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_applications_job ON applications (job_id);
CREATE INDEX IF NOT EXISTS ix_applications_status     ON applications (status);
CREATE INDEX IF NOT EXISTS ix_applications_run        ON applications (run_id);
CREATE INDEX IF NOT EXISTS ix_applications_submitted  ON applications (submitted_at DESC)
    WHERE submitted_at IS NOT NULL;

-- ── Application answers ────────────────────────────────────────────────────
-- Every screening question and where its answer came from. `source` is the
-- audit trail that proves a number was taken from the profile rather than
-- invented by a model.
CREATE TABLE IF NOT EXISTS application_answers (
    id                  BIGSERIAL PRIMARY KEY,
    application_id      BIGINT      NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    question            TEXT        NOT NULL,
    question_type       TEXT        NOT NULL DEFAULT 'text'
        CHECK (question_type IN ('text','numeric','boolean','select','radio','checkbox','file','unknown')),
    answer              TEXT,
    source              TEXT        NOT NULL DEFAULT 'unknown'
        CHECK (source IN ('profile','resume','ai','manual','default','unknown')),
    confidence          NUMERIC(3,2) CHECK (confidence BETWEEN 0 AND 1),
    requires_review     BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_answers_application ON application_answers (application_id);
-- Feeds the "questions I have been asked before" learning loop.
CREATE INDEX IF NOT EXISTS ix_answers_question    ON application_answers (lower(question));

-- ── Events (audit log) ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS events (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              BIGINT      REFERENCES runs(id) ON DELETE CASCADE,
    application_id      BIGINT      REFERENCES applications(id) ON DELETE CASCADE,
    job_id              BIGINT      REFERENCES jobs(id) ON DELETE SET NULL,
    level               TEXT        NOT NULL DEFAULT 'info'
        CHECK (level IN ('debug','info','warning','error','critical')),
    event_type          TEXT        NOT NULL,
    message             TEXT        NOT NULL DEFAULT '',
    data                JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_events_run        ON events (run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_events_type       ON events (event_type);
CREATE INDEX IF NOT EXISTS ix_events_created_at ON events (created_at DESC);

-- ── Notification log ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS notification_log (
    id              BIGSERIAL PRIMARY KEY,
    channel         TEXT        NOT NULL,
    event_type      TEXT        NOT NULL DEFAULT '',
    subject         TEXT        NOT NULL DEFAULT '',
    status          TEXT        NOT NULL DEFAULT 'sent' CHECK (status IN ('sent','failed')),
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_notification_log_created ON notification_log (created_at DESC);

-- ── Key/value app state ────────────────────────────────────────────────────
-- Session validity, last-scan cursors, circuit-breaker state.
CREATE TABLE IF NOT EXISTS app_state (
    key         TEXT PRIMARY KEY,
    value       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── updated_at maintenance ─────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY['resumes','search_profiles','runs','applications']
    LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_%1$s_updated_at ON %1$s;
             CREATE TRIGGER trg_%1$s_updated_at BEFORE UPDATE ON %1$s
             FOR EACH ROW EXECUTE FUNCTION set_updated_at();', t);
    END LOOP;
END $$;

-- ── Reporting view ─────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_application_summary AS
SELECT
    a.id                AS application_id,
    a.status,
    a.score,
    a.submitted_at,
    a.created_at,
    j.title,
    j.company,
    j.location,
    j.url,
    r.label             AS resume_label,
    a.run_id
FROM applications a
JOIN jobs j    ON j.id = a.job_id
LEFT JOIN resumes r ON r.id = a.resume_id;

INSERT INTO schema_migrations (version) VALUES ('001_initial_schema')
ON CONFLICT (version) DO NOTHING;
