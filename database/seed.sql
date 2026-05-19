INSERT INTO model_configs (
    name, k_value, method, distance_metric, dataset_version, active
) VALUES (
    'Default LSH kNN k=3', 3, 'lsh', 'euclidean', 1, TRUE
)
ON CONFLICT DO NOTHING;
