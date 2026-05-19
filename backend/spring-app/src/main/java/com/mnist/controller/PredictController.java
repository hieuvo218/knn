package com.mnist.controller;

import com.mnist.dto.PredictRequest;
import com.mnist.dto.PredictResponse;
import com.mnist.service.MlClient;
import com.mnist.service.ModelService;
import com.mnist.service.PixelUtils;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.support.GeneratedKeyHolder;
import org.springframework.jdbc.support.KeyHolder;
import org.springframework.web.bind.annotation.*;

import java.sql.PreparedStatement;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class PredictController {
    private final JdbcTemplate jdbcTemplate;
    private final PixelUtils pixelUtils;
    private final ModelService modelService;
    private final MlClient mlClient;

    public PredictController(JdbcTemplate jdbcTemplate, PixelUtils pixelUtils, ModelService modelService, MlClient mlClient) {
        this.jdbcTemplate = jdbcTemplate;
        this.pixelUtils = pixelUtils;
        this.modelService = modelService;
        this.mlClient = mlClient;
    }

    @GetMapping("/predict")
    public PredictResponse predictGet(@RequestParam String pixels) {
        return runPrediction(pixelUtils.parsePixels(pixels));
    }

    @PostMapping("/predict")
    public PredictResponse predictPost(@RequestBody PredictRequest request) {
        pixelUtils.validatePixels(request.pixels());
        return runPrediction(request.pixels());
    }

    private PredictResponse runPrediction(List<Integer> pixels) {
        Map<String, Object> model = modelService.activeModel();
        long modelId = ((Number) model.get("id")).longValue();
        int k = ((Number) model.get("k")).intValue();
        String method = (String) model.get("method");

        Map<String, Object> ml = mlClient.predict(pixels, k, method);
        int predictedLabel = ((Number) ml.get("predictedLabel")).intValue();
        double confidence = ((Number) ml.get("confidence")).doubleValue();
        int responseTimeMs = ((Number) ml.get("latencyMs")).intValue();
        long datasetVersion = ((Number) ml.get("datasetVersion")).longValue();
        int sampleCount = ((Number) ml.get("sampleCount")).intValue();

        KeyHolder keyHolder = new GeneratedKeyHolder();
        jdbcTemplate.update(connection -> {
            PreparedStatement ps = connection.prepareStatement("""
                INSERT INTO predictions(input_pixels, predicted_label, confidence, model_id, response_time_ms)
                VALUES (?, ?, ?, ?, ?)
            """, new String[]{"id"});
            ps.setArray(1, pixelUtils.toSqlSmallIntArray(connection, pixels));
            ps.setInt(2, predictedLabel);
            ps.setDouble(3, confidence);
            ps.setLong(4, modelId);
            ps.setInt(5, responseTimeMs);
            return ps;
        }, keyHolder);

        long predictionId = extractGeneratedId(keyHolder);
        return new PredictResponse(predictionId, predictedLabel, confidence, responseTimeMs, k, method, datasetVersion, sampleCount);
    }

    @PostMapping("/predictions/{id}/confirm")
    public Map<String, Object> confirmPrediction(@PathVariable long id) {
        int updated = jdbcTemplate.update("""
            UPDATE predictions
            SET accepted = TRUE, corrected_label = NULL, updated_at = NOW()
            WHERE id = ?
        """, id);
        return Map.of("updated", updated, "predictionId", id, "accepted", true);
    }

    private long extractGeneratedId(KeyHolder keyHolder) {
        Number key = keyHolder.getKey();
        if (key != null) {
            return key.longValue();
        }

        Map<String, Object> keys = keyHolder.getKeys();
        if (keys != null && keys.get("id") instanceof Number id) {
            return id.longValue();
        }

        throw new IllegalStateException("Could not resolve generated id from insert result");
    }
}
