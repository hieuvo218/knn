package com.mnist.dto;

import java.util.List;

public record PredictRequest(List<Integer> pixels) {}
