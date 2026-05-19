package com.mnist.service;

import com.mnist.dto.TuneRequest;
import com.mnist.dto.TuneResponse;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class MlClient {
    private final RestTemplate restTemplate;
    private final String mlServiceUrl;

    public MlClient(RestTemplate restTemplate, @Qualifier("mlServiceUrl") String mlServiceUrl) {
        this.restTemplate = restTemplate;
        this.mlServiceUrl = mlServiceUrl;
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> predict(List<Integer> pixels, int k, String method) {
        Map<String, Object> body = new HashMap<>();
        body.put("pixels", pixels);
        body.put("k", k);
        body.put("method", method);
        return restTemplate.postForObject(mlServiceUrl + "/predict", body, Map.class);
    }

    public TuneResponse tune(TuneRequest request) {
        return restTemplate.postForObject(mlServiceUrl + "/tune", request, TuneResponse.class);
    }
}
