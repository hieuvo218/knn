package com.mnist.dto;

import java.util.List;

public record TuneRequest(
        Integer sampleCount,
        String method,
        List<Integer> kValues
) {}
