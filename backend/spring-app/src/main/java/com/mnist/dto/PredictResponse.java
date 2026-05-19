package com.mnist.dto;

public record PredictResponse(
        long predictionId,
        int predictedLabel,
        double confidence,
        int responseTimeMs,
        int k,
        String method,
        long datasetVersion,
        int sampleCount
) {}
