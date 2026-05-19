package com.mnist.service;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
public class ModelService {
    private final JdbcTemplate jdbcTemplate;

    public ModelService(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    public Map<String, Object> activeModel() {
        return jdbcTemplate.queryForObject("""
            SELECT id, name, k_value, method, distance_metric, dataset_version,
                   COALESCE(accuracy, 0) AS accuracy,
                   COALESCE(f1_score, 0) AS f1_score,
                   COALESCE(avg_latency_ms, 0) AS avg_latency_ms
            FROM model_configs
            WHERE active = TRUE
            ORDER BY updated_at DESC
            LIMIT 1
        """, (rs, rowNum) -> {
            Map<String, Object> m = new HashMap<>();
            m.put("id", rs.getLong("id"));
            m.put("name", rs.getString("name"));
            m.put("k", rs.getInt("k_value"));
            m.put("method", rs.getString("method"));
            m.put("distanceMetric", rs.getString("distance_metric"));
            m.put("datasetVersion", rs.getLong("dataset_version"));
            m.put("accuracy", rs.getDouble("accuracy"));
            m.put("f1Score", rs.getDouble("f1_score"));
            m.put("avgLatencyMs", rs.getDouble("avg_latency_ms"));
            return m;
        });
    }

    public long currentDatasetVersion() {
        return jdbcTemplate.queryForObject("SELECT version FROM dataset_state WHERE id = 1", Long.class);
    }

    public void bumpDatasetVersion() {
        jdbcTemplate.update("UPDATE dataset_state SET version = version + 1, updated_at = NOW() WHERE id = 1");
    }
}
