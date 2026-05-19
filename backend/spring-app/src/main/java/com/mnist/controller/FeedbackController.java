package com.mnist.controller;

import com.mnist.dto.FeedbackRequest;
import com.mnist.dto.FeedbackResponse;
import com.mnist.service.ModelService;
import com.mnist.service.PixelUtils;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.support.GeneratedKeyHolder;
import org.springframework.jdbc.support.KeyHolder;
import org.springframework.web.bind.annotation.*;

import java.sql.PreparedStatement;
import java.util.*;

@RestController
@RequestMapping("/api")
public class FeedbackController {
    private final JdbcTemplate jdbcTemplate;
    private final PixelUtils pixelUtils;
    private final ModelService modelService;

    public FeedbackController(JdbcTemplate jdbcTemplate, PixelUtils pixelUtils, ModelService modelService) {
        this.jdbcTemplate = jdbcTemplate;
        this.pixelUtils = pixelUtils;
        this.modelService = modelService;
    }

    @PostMapping("/feedback")
    public FeedbackResponse createFeedback(@RequestBody FeedbackRequest request) {
        pixelUtils.validatePixels(request.pixels());
        pixelUtils.validateLabel(request.trueLabel());

        Integer predictedLabel = request.predictedLabel();
        if (predictedLabel != null) pixelUtils.validateLabel(predictedLabel);

        KeyHolder keyHolder = new GeneratedKeyHolder();
        jdbcTemplate.update(connection -> {
            PreparedStatement ps = connection.prepareStatement("""
                INSERT INTO feedback_samples(pixels, predicted_label, true_label, prediction_id, status)
                VALUES (?, ?, ?, ?, 'pending')
            """, new String[]{"id"});
            ps.setArray(1, pixelUtils.toSqlSmallIntArray(connection, request.pixels()));
            if (predictedLabel == null) ps.setNull(2, java.sql.Types.SMALLINT); else ps.setInt(2, predictedLabel);
            ps.setInt(3, request.trueLabel());
            if (request.predictionId() == null) ps.setNull(4, java.sql.Types.BIGINT); else ps.setLong(4, request.predictionId());
            return ps;
        }, keyHolder);
        long feedbackId = extractGeneratedId(keyHolder);

        if (request.predictionId() != null) {
            jdbcTemplate.update("""
                UPDATE predictions
                SET accepted = FALSE, corrected_label = ?, feedback_id = ?, updated_at = NOW()
                WHERE id = ?
            """, request.trueLabel(), feedbackId, request.predictionId());
        }

        return new FeedbackResponse(feedbackId, "pending", request.trueLabel());
    }

    @GetMapping("/feedback")
    public List<Map<String, Object>> listFeedback(@RequestParam(defaultValue = "pending") String status) {
        return jdbcTemplate.query("""
            SELECT id, pixels, predicted_label, true_label, prediction_id, status, created_at, reviewed_at
            FROM feedback_samples
            WHERE status = ?
            ORDER BY true_label ASC, created_at DESC
        """, (rs, rowNum) -> {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", rs.getLong("id"));
            m.put("pixels", arrayToList(rs.getArray("pixels")));
            Object predicted = rs.getObject("predicted_label");
            m.put("predictedLabel", predicted == null ? null : ((Number) predicted).intValue());
            m.put("trueLabel", rs.getInt("true_label"));
            Object predictionId = rs.getObject("prediction_id");
            m.put("predictionId", predictionId == null ? null : ((Number) predictionId).longValue());
            m.put("status", rs.getString("status"));
            m.put("createdAt", rs.getTimestamp("created_at").toInstant().toString());
            Object reviewedAt = rs.getTimestamp("reviewed_at");
            m.put("reviewedAt", reviewedAt == null ? null : ((java.sql.Timestamp) reviewedAt).toInstant().toString());
            return m;
        }, status);
    }

    @PostMapping("/feedback/{id}/accept")
    public Map<String, Object> acceptFeedback(@PathVariable long id) {
        Map<String, Object> feedback = jdbcTemplate.queryForObject("""
            SELECT id, pixels, true_label, status
            FROM feedback_samples
            WHERE id = ?
        """, (rs, rowNum) -> {
            Map<String, Object> m = new HashMap<>();
            m.put("id", rs.getLong("id"));
            m.put("pixels", arrayToList(rs.getArray("pixels")));
            m.put("trueLabel", rs.getInt("true_label"));
            m.put("status", rs.getString("status"));
            return m;
        }, id);

        if (!"pending".equals(feedback.get("status"))) {
            throw new IllegalArgumentException("Only pending feedback can be accepted");
        }

        @SuppressWarnings("unchecked")
        List<Integer> pixels = (List<Integer>) feedback.get("pixels");
        int trueLabel = (Integer) feedback.get("trueLabel");

        KeyHolder sampleKey = new GeneratedKeyHolder();
        jdbcTemplate.update(connection -> {
            PreparedStatement ps = connection.prepareStatement("""
                INSERT INTO digit_samples(pixels, label, source, status, accepted_at)
                VALUES (?, ?, 'feedback', 'accepted', NOW())
            """, new String[]{"id"});
            ps.setArray(1, pixelUtils.toSqlSmallIntArray(connection, pixels));
            ps.setInt(2, trueLabel);
            return ps;
        }, sampleKey);

        jdbcTemplate.update("UPDATE feedback_samples SET status='accepted', reviewed_at=NOW() WHERE id=?", id);
        modelService.bumpDatasetVersion();

        return Map.of("feedbackId", id, "status", "accepted", "sampleId", extractGeneratedId(sampleKey), "datasetVersion", modelService.currentDatasetVersion());
    }

    @PostMapping("/feedback/{id}/reject")
    public Map<String, Object> rejectFeedback(@PathVariable long id) {
        int updated = jdbcTemplate.update("""
            UPDATE feedback_samples
            SET status='rejected', reviewed_at=NOW()
            WHERE id=? AND status='pending'
        """, id);
        return Map.of("feedbackId", id, "status", "rejected", "updated", updated);
    }

    private List<Integer> arrayToList(java.sql.Array array) throws java.sql.SQLException {
        Object raw = array.getArray();
        List<Integer> out = new ArrayList<>();
        if (raw instanceof Short[] shorts) {
            for (Short s : shorts) out.add(s.intValue());
        } else if (raw instanceof Integer[] ints) {
            out.addAll(Arrays.asList(ints));
        } else if (raw instanceof Object[] objs) {
            for (Object o : objs) out.add(((Number) o).intValue());
        }
        return out;
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
