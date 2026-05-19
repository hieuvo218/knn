package com.mnist.dto;

public record FeedbackResponse(
        long feedbackId,
        String status,
        Integer trueLabel
) {}
