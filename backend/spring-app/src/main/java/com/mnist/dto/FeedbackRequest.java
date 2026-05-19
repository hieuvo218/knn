package com.mnist.dto;

import java.util.List;

public record FeedbackRequest(
        List<Integer> pixels,
        Integer predictionId,
        Integer predictedLabel,
        Integer trueLabel
) {}
