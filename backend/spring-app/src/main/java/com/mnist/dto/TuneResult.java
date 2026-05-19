package com.mnist.dto;

public record TuneResult(
        int rank,
        int k,
        String method,
        double accuracy,
        double f1Score,
        double avgLatencyMs,
        int trainingSamples,
        int evaluatedSamples,
        long datasetVersion
) {}
