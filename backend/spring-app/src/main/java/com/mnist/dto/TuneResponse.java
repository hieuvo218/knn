package com.mnist.dto;

import java.util.List;

public record TuneResponse(
        String jobId,
        long datasetVersion,
        int sampleCount,
        String method,
        List<TuneResult> topResults
) {}
