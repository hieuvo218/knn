package com.mnist.controller;

import com.mnist.dto.*;
import com.mnist.service.MlClient;
import com.mnist.service.ModelService;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.*;

import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api")
public class ModelController {
    private final JdbcTemplate jdbcTemplate;
    private final MlClient mlClient;
    private final ModelService modelService;

    public ModelController(JdbcTemplate jdbcTemplate, MlClient mlClient, ModelService modelService) {
        this.jdbcTemplate = jdbcTemplate;
        this.mlClient = mlClient;
        this.modelService = modelService;
    }

    @GetMapping("/model/active")
    public Map<String, Object> activeModel() {
        return modelService.activeModel();
    }

    @PostMapping("/tune")
    public TuneResponse tune(@RequestBody TuneRequest request) {
        TuneRequest fixed = normalizeTuneRequest(request);
        TuneResponse response = mlClient.tune(fixed);

        jdbcTemplate.update("""
            INSERT INTO tuning_jobs(id, sample_count, method, k_values, dataset_version)
            VALUES (?, ?, ?, ?, ?)
        """, response.jobId(), response.sampleCount(), response.method(),
                fixed.kValues().stream().map(String::valueOf).collect(Collectors.joining(",")),
                response.datasetVersion());

        for (TuneResult result : response.topResults()) {
            jdbcTemplate.update("""
                INSERT INTO tuning_results(job_id, k_value, method, accuracy, f1_score, avg_latency_ms,
                                           training_samples, evaluated_samples, dataset_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, response.jobId(), result.k(), result.method(), result.accuracy(), result.f1Score(),
                    result.avgLatencyMs(), result.trainingSamples(), result.evaluatedSamples(), result.datasetVersion());
        }

        return response;
    }

    @PostMapping("/tune/{jobId}/activate")
    public Map<String, Object> activateTune(@PathVariable String jobId, @RequestBody ActivateTuneRequest request) {
        int k = request.k() == null ? 3 : request.k();
        String method = request.method() == null ? "kd_tree" : request.method();
        if (!method.equals("kd_tree") && !method.equals("lsh")) {
            throw new IllegalArgumentException("method must be kd_tree or lsh");
        }

        Map<String, Object> result = jdbcTemplate.queryForObject("""
            SELECT k_value, method, accuracy, f1_score, avg_latency_ms, dataset_version
            FROM tuning_results
            WHERE job_id = ? AND k_value = ? AND method = ?
            ORDER BY accuracy DESC, avg_latency_ms ASC, f1_score DESC
            LIMIT 1
        """, (rs, rowNum) -> {
            Map<String, Object> m = new HashMap<>();
            m.put("k", rs.getInt("k_value"));
            m.put("method", rs.getString("method"));
            m.put("accuracy", rs.getDouble("accuracy"));
            m.put("f1Score", rs.getDouble("f1_score"));
            m.put("avgLatencyMs", rs.getDouble("avg_latency_ms"));
            m.put("datasetVersion", rs.getLong("dataset_version"));
            return m;
        }, jobId, k, method);

        jdbcTemplate.update("UPDATE model_configs SET active=FALSE WHERE active=TRUE");
        jdbcTemplate.update("""
            INSERT INTO model_configs(name, k_value, method, distance_metric, dataset_version,
                                      accuracy, f1_score, avg_latency_ms, active)
            VALUES (?, ?, ?, 'euclidean', ?, ?, ?, ?, TRUE)
        """, "Tuned kNN k=" + k + " " + method, k, method,
                result.get("datasetVersion"), result.get("accuracy"), result.get("f1Score"), result.get("avgLatencyMs"));

        return Map.of("status", "activated", "jobId", jobId, "k", k, "method", method);
    }

    @GetMapping("/model/dashboard")
    public Map<String, Object> dashboard() {
        Map<String, Object> active = modelService.activeModel();
        long totalPredictions = jdbcTemplate.queryForObject("SELECT COUNT(*) FROM predictions", Long.class);
        double avgResponseTime = Optional.ofNullable(jdbcTemplate.queryForObject(
                "SELECT COALESCE(AVG(response_time_ms), 0) FROM predictions", Double.class)).orElse(0.0);
        long acceptedSamples = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM digit_samples WHERE status='accepted' AND deleted=FALSE", Long.class);
        long pendingFeedback = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM feedback_samples WHERE status='pending'", Long.class);
        long datasetVersion = modelService.currentDatasetVersion();

        List<Map<String, Object>> labeledPredictions = jdbcTemplate.query("""
            SELECT predicted_label,
                   CASE WHEN accepted = TRUE THEN predicted_label ELSE corrected_label END AS true_label
            FROM predictions
            WHERE accepted IS NOT NULL
              AND (accepted = TRUE OR corrected_label IS NOT NULL)
        """, (rs, rowNum) -> {
            Map<String, Object> m = new HashMap<>();
            m.put("predicted", rs.getInt("predicted_label"));
            m.put("true", rs.getInt("true_label"));
            return m;
        });

        int[][] confusion = new int[10][10];
        int correct = 0;
        for (Map<String, Object> row : labeledPredictions) {
            int trueLabel = ((Number) row.get("true")).intValue();
            int pred = ((Number) row.get("predicted")).intValue();
            confusion[trueLabel][pred]++;
            if (trueLabel == pred) correct++;
        }

        double accuracy = labeledPredictions.isEmpty() ? 0.0 : (double) correct / labeledPredictions.size();
        double f1 = macroF1(confusion);

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("activeModel", active);
        response.put("datasetVersion", datasetVersion);
        response.put("acceptedSamples", acceptedSamples);
        response.put("pendingFeedback", pendingFeedback);
        response.put("totalPredictions", totalPredictions);
        response.put("accuracy", accuracy);
        response.put("f1Score", f1);
        response.put("avgResponseTimeMs", avgResponseTime);
        response.put("confusionMatrix", confusion);
        return response;
    }

    private TuneRequest normalizeTuneRequest(TuneRequest request) {
        int sampleCount = request.sampleCount() == null ? 500 : Math.max(20, request.sampleCount());
        String method = request.method() == null ? "kd_tree" : request.method();
        if (!method.equals("kd_tree") && !method.equals("lsh")) {
            throw new IllegalArgumentException("method must be kd_tree or lsh");
        }
        List<Integer> kValues = request.kValues();
        if (kValues == null || kValues.isEmpty()) kValues = List.of(1, 3, 5, 7);
        kValues = kValues.stream().filter(k -> k != null && k > 0).distinct().sorted().toList();
        return new TuneRequest(sampleCount, method, kValues);
    }

    private double macroF1(int[][] matrix) {
        double total = 0.0;
        for (int label = 0; label < 10; label++) {
            int tp = matrix[label][label];
            int fp = 0;
            int fn = 0;
            for (int i = 0; i < 10; i++) {
                if (i != label) {
                    fp += matrix[i][label];
                    fn += matrix[label][i];
                }
            }
            double precision = tp + fp == 0 ? 0.0 : (double) tp / (tp + fp);
            double recall = tp + fn == 0 ? 0.0 : (double) tp / (tp + fn);
            double f1 = precision + recall == 0 ? 0.0 : 2 * precision * recall / (precision + recall);
            total += f1;
        }
        return total / 10.0;
    }
}
