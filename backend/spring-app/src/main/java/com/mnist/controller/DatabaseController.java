package com.mnist.controller;

import com.mnist.dto.UpdateSampleRequest;
import com.mnist.service.ModelService;
import com.mnist.service.PixelUtils;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping("/api/database")
public class DatabaseController {
    private final JdbcTemplate jdbcTemplate;
    private final PixelUtils pixelUtils;
    private final ModelService modelService;

    public DatabaseController(JdbcTemplate jdbcTemplate, PixelUtils pixelUtils, ModelService modelService) {
        this.jdbcTemplate = jdbcTemplate;
        this.pixelUtils = pixelUtils;
        this.modelService = modelService;
    }

    @GetMapping("/stats")
    public Map<String, Object> stats() {
        long total = jdbcTemplate.queryForObject("SELECT COUNT(*) FROM digit_samples WHERE deleted=FALSE", Long.class);
        long accepted = jdbcTemplate.queryForObject("SELECT COUNT(*) FROM digit_samples WHERE deleted=FALSE AND status='accepted'", Long.class);
        long pendingFeedback = jdbcTemplate.queryForObject("SELECT COUNT(*) FROM feedback_samples WHERE status='pending'", Long.class);
        long version = modelService.currentDatasetVersion();

        List<Map<String, Object>> distribution = jdbcTemplate.query("""
            SELECT label, COUNT(*) AS count
            FROM digit_samples
            WHERE deleted=FALSE AND status='accepted'
            GROUP BY label
            ORDER BY label
        """, (rs, rowNum) -> Map.of("label", rs.getInt("label"), "count", rs.getLong("count")));

        return Map.of(
                "totalSamples", total,
                "acceptedSamples", accepted,
                "pendingFeedback", pendingFeedback,
                "datasetVersion", version,
                "distribution", distribution
        );
    }

    @GetMapping("/samples")
    public Map<String, Object> samples(
            @RequestParam(required = false) Long id,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String source,
            @RequestParam(defaultValue = "latest") String order,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size
    ) {
        page = Math.max(0, page);
        size = Math.min(Math.max(1, size), 100);
        int offset = page * size;

        String fromClause = """
            FROM (
                SELECT id, pixels, label, source, status, created_at, updated_at, 'digit' AS row_type
                FROM digit_samples
                WHERE deleted=FALSE

                UNION ALL

                SELECT id, pixels, true_label AS label, 'feedback_submission' AS source, status,
                       created_at, COALESCE(reviewed_at, created_at) AS updated_at, 'feedback' AS row_type
                FROM feedback_samples
            ) merged
        """;

        StringBuilder where = new StringBuilder(" WHERE 1=1 ");
        List<Object> params = new ArrayList<>();
        if (id != null) { where.append(" AND id=? "); params.add(id); }
        if (status != null && !status.isBlank()) { where.append(" AND status=? "); params.add(status); }
        if (source != null && !source.isBlank()) { where.append(" AND source=? "); params.add(source); }

        String normalizedOrder = (order == null) ? "latest" : order.trim().toLowerCase(Locale.ROOT);
        String orderBy = switch (normalizedOrder) {
            case "oldest", "oldest_first" -> " ORDER BY created_at ASC, id ASC ";
            case "id_asc" -> " ORDER BY id ASC ";
            case "id_desc" -> " ORDER BY id DESC ";
            case "newest", "latest" -> " ORDER BY created_at DESC, id DESC ";
            default -> " ORDER BY created_at DESC, id DESC ";
        };

        Long total = jdbcTemplate.queryForObject("SELECT COUNT(*) " + fromClause + where, Long.class, params.toArray());
        params.add(size);
        params.add(offset);

        List<Map<String, Object>> rows = jdbcTemplate.query("""
            SELECT id, pixels, label, source, status, created_at, updated_at, row_type
            """ + fromClause + where + orderBy + " LIMIT ? OFFSET ?", (rs, rowNum) -> {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", rs.getLong("id"));
            m.put("pixels", arrayToList(rs.getArray("pixels")));
            m.put("label", rs.getInt("label"));
            m.put("source", rs.getString("source"));
            m.put("status", rs.getString("status"));
            m.put("createdAt", rs.getTimestamp("created_at").toInstant().toString());
            m.put("updatedAt", rs.getTimestamp("updated_at").toInstant().toString());
            m.put("rowType", rs.getString("row_type"));
            return m;
        }, params.toArray());

        return Map.of("page", page, "size", size, "total", total, "rows", rows);
    }

    @PutMapping("/samples/{id}")
    public Map<String, Object> updateLabel(@PathVariable long id, @RequestBody UpdateSampleRequest request) {
        pixelUtils.validateLabel(request.label());
        String status = jdbcTemplate.queryForObject("SELECT status FROM digit_samples WHERE id=? AND deleted=FALSE", String.class, id);
        int updated = jdbcTemplate.update("UPDATE digit_samples SET label=? WHERE id=? AND deleted=FALSE", request.label(), id);
        if ("accepted".equals(status) && updated > 0) modelService.bumpDatasetVersion();
        return Map.of("sampleId", id, "updated", updated, "datasetVersion", modelService.currentDatasetVersion());
    }

    @DeleteMapping("/samples/{id}")
    public Map<String, Object> deleteSample(@PathVariable long id) {
        String status = jdbcTemplate.queryForObject("SELECT status FROM digit_samples WHERE id=? AND deleted=FALSE", String.class, id);
        int updated = jdbcTemplate.update("UPDATE digit_samples SET deleted=TRUE WHERE id=? AND deleted=FALSE", id);
        if ("accepted".equals(status) && updated > 0) modelService.bumpDatasetVersion();
        return Map.of("sampleId", id, "deleted", updated > 0, "datasetVersion", modelService.currentDatasetVersion());
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
}
