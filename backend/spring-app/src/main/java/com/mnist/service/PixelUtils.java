package com.mnist.service;

import org.springframework.stereotype.Component;

import java.sql.Connection;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

@Component
public class PixelUtils {
    public List<Integer> parsePixels(String pixelsText) {
        if (pixelsText == null || pixelsText.isBlank()) {
            throw new IllegalArgumentException("pixels query parameter is required");
        }
        String[] parts = pixelsText.split(",");
        List<Integer> pixels = new ArrayList<>(parts.length);
        for (String part : parts) {
            pixels.add(Integer.parseInt(part.trim()));
        }
        validatePixels(pixels);
        return pixels;
    }

    public void validatePixels(List<Integer> pixels) {
        if (pixels == null || pixels.size() != 784) {
            throw new IllegalArgumentException("pixels must contain exactly 784 integers");
        }
        for (Integer pixel : pixels) {
            if (pixel == null || pixel < 0 || pixel > 255) {
                throw new IllegalArgumentException("pixels must be integers in [0, 255]");
            }
        }
    }

    public void validateLabel(Integer label) {
        if (label == null || label < 0 || label > 9) {
            throw new IllegalArgumentException("label must be an integer from 0 to 9");
        }
    }

    public Short[] toShortArray(List<Integer> pixels) {
        validatePixels(pixels);
        Short[] out = new Short[pixels.size()];
        for (int i = 0; i < pixels.size(); i++) {
            out[i] = pixels.get(i).shortValue();
        }
        return out;
    }

    public java.sql.Array toSqlSmallIntArray(Connection connection, List<Integer> pixels) throws SQLException {
        return connection.createArrayOf("smallint", toShortArray(pixels));
    }
}
