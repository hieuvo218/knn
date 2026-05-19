const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080/api';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    throw new Error(data?.message || data?.error || `HTTP ${response.status}`);
  }
  return data;
}

export const api = {
  health: () => request('/health'),
  predict: (pixels) => request('/predict', { method: 'POST', body: JSON.stringify({ pixels }) }),
  confirmPrediction: (id) => request(`/predictions/${id}/confirm`, { method: 'POST' }),
  createFeedback: ({ pixels, predictionId, predictedLabel, trueLabel }) =>
    request('/feedback', { method: 'POST', body: JSON.stringify({ pixels, predictionId, predictedLabel, trueLabel }) }),
  listFeedback: (status = 'pending') => request(`/feedback?status=${encodeURIComponent(status)}`),
  acceptFeedback: (id) => request(`/feedback/${id}/accept`, { method: 'POST' }),
  rejectFeedback: (id) => request(`/feedback/${id}/reject`, { method: 'POST' }),
  dashboard: () => request('/model/dashboard'),
  activeModel: () => request('/model/active'),
  tune: ({ sampleCount, method, kValues }) =>
    request('/tune', { method: 'POST', body: JSON.stringify({ sampleCount, method, kValues }) }),
  activateTune: ({ jobId, k, method }) =>
    request(`/tune/${jobId}/activate`, { method: 'POST', body: JSON.stringify({ k, method }) }),
  dbStats: () => request('/database/stats'),
  dbSamples: ({ page = 0, size = 20, id = '', status = '', source = '', order = 'latest' } = {}) => {
    const params = new URLSearchParams({ page, size });
    if (id !== '') params.set('id', id);
    if (status) params.set('status', status);
    if (source) params.set('source', source);
    if (order) params.set('order', order);
    return request(`/database/samples?${params.toString()}`);
  },
  updateSample: (id, label) => request(`/database/samples/${id}`, { method: 'PUT', body: JSON.stringify({ label }) }),
  deleteSample: (id) => request(`/database/samples/${id}`, { method: 'DELETE' }),
};
