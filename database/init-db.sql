CREATE TABLE IF NOT EXISTS dataset_state (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    version BIGINT NOT NULL DEFAULT 1,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

INSERT INTO dataset_state (id, version) VALUES (1, 1)
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS digit_samples (
    id BIGSERIAL PRIMARY KEY,
    pixels SMALLINT[] NOT NULL,
    label SMALLINT NOT NULL CHECK (label BETWEEN 0 AND 9),
    source VARCHAR(40) NOT NULL DEFAULT 'mnist',
    status VARCHAR(20) NOT NULL DEFAULT 'accepted' CHECK (status IN ('accepted', 'pending', 'rejected')),
    deleted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    accepted_at TIMESTAMP,
    CHECK (array_length(pixels, 1) = 784)
);

CREATE TABLE IF NOT EXISTS model_configs (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    k_value INTEGER NOT NULL CHECK (k_value > 0),
    method VARCHAR(20) NOT NULL CHECK (method IN ('kd_tree', 'lsh')),
    distance_metric VARCHAR(30) NOT NULL DEFAULT 'euclidean',
    dataset_version BIGINT NOT NULL DEFAULT 1,
    accuracy DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    avg_latency_ms DOUBLE PRECISION,
    active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_model
ON model_configs(active)
WHERE active = TRUE;

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    input_pixels SMALLINT[] NOT NULL,
    predicted_label SMALLINT NOT NULL CHECK (predicted_label BETWEEN 0 AND 9),
    confidence DOUBLE PRECISION,
    model_id BIGINT REFERENCES model_configs(id),
    response_time_ms INTEGER,
    accepted BOOLEAN,
    corrected_label SMALLINT CHECK (corrected_label IS NULL OR corrected_label BETWEEN 0 AND 9),
    feedback_id BIGINT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CHECK (array_length(input_pixels, 1) = 784)
);

CREATE TABLE IF NOT EXISTS feedback_samples (
    id BIGSERIAL PRIMARY KEY,
    pixels SMALLINT[] NOT NULL,
    predicted_label SMALLINT CHECK (predicted_label IS NULL OR predicted_label BETWEEN 0 AND 9),
    true_label SMALLINT NOT NULL CHECK (true_label BETWEEN 0 AND 9),
    prediction_id BIGINT REFERENCES predictions(id),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'rejected')),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    reviewed_at TIMESTAMP,
    CHECK (array_length(pixels, 1) = 784)
);

ALTER TABLE predictions
ADD CONSTRAINT fk_predictions_feedback
FOREIGN KEY (feedback_id) REFERENCES feedback_samples(id)
DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE IF NOT EXISTS tuning_jobs (
    id VARCHAR(80) PRIMARY KEY,
    sample_count INTEGER NOT NULL,
    method VARCHAR(20) NOT NULL CHECK (method IN ('kd_tree', 'lsh')),
    k_values TEXT NOT NULL,
    dataset_version BIGINT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tuning_results (
    id BIGSERIAL PRIMARY KEY,
    job_id VARCHAR(80) NOT NULL REFERENCES tuning_jobs(id) ON DELETE CASCADE,
    k_value INTEGER NOT NULL,
    method VARCHAR(20) NOT NULL CHECK (method IN ('kd_tree', 'lsh')),
    accuracy DOUBLE PRECISION NOT NULL,
    f1_score DOUBLE PRECISION NOT NULL,
    avg_latency_ms DOUBLE PRECISION NOT NULL,
    training_samples INTEGER NOT NULL,
    evaluated_samples INTEGER NOT NULL,
    dataset_version BIGINT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS digit_samples_touch ON digit_samples;
CREATE TRIGGER digit_samples_touch BEFORE UPDATE ON digit_samples
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS model_configs_touch ON model_configs;
CREATE TRIGGER model_configs_touch BEFORE UPDATE ON model_configs
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS predictions_touch ON predictions;
CREATE TRIGGER predictions_touch BEFORE UPDATE ON predictions
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

CREATE INDEX IF NOT EXISTS idx_digit_samples_label ON digit_samples(label);
CREATE INDEX IF NOT EXISTS idx_digit_samples_status ON digit_samples(status);
CREATE INDEX IF NOT EXISTS idx_digit_samples_active ON digit_samples(status, deleted) WHERE status='accepted' AND deleted=FALSE;
CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback_samples(status);
CREATE INDEX IF NOT EXISTS idx_feedback_true_label ON feedback_samples(true_label);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tuning_results_job ON tuning_results(job_id);
