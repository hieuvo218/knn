import React, { useState } from 'react';
import { api } from '../api.js';
import DigitCanvas from '../components/DigitCanvas.jsx';
import DigitPreview from '../components/DigitPreview.jsx';

export default function Home() {
  const [pixels, setPixels] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [trueLabel, setTrueLabel] = useState('');
  const [confirmed, setConfirmed] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [status, setStatus] = useState('Draw a digit. Prediction runs 1s after you stop drawing.');
  const [loading, setLoading] = useState(false);

  async function handlePixelsChange(nextPixels) {
    setPixels(nextPixels);
    setPrediction(null);
    setTrueLabel('');
    setConfirmed(false);
    setFeedbackSubmitted(false);
    if (!nextPixels) {
      setStatus('Canvas cleared.');
      return;
    }
    const ink = nextPixels.reduce((sum, v) => sum + v, 0);
    if (ink < 300) {
      setStatus('Draw a bit more ink before predicting.');
      return;
    }
    setLoading(true);
    setStatus('Predicting...');
    try {
      const result = await api.predict(nextPixels);
      setPrediction(result);
      setConfirmed(false);
      setStatus('Prediction ready.');
    } catch (err) {
      setStatus(`Prediction failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function submitFeedback() {
    if (!pixels || !prediction || trueLabel === '') return;
    try {
      const res = await api.createFeedback({
        pixels,
        predictionId: prediction.predictionId,
        predictedLabel: prediction.predictedLabel,
        trueLabel: Number(trueLabel),
      });
      setStatus(`Feedback submitted as pending. Feedback id: ${res.feedbackId}`);
      setFeedbackSubmitted(true);
    } catch (err) {
      setStatus(`Feedback failed: ${err.message}`);
    }
  }

  async function confirmCorrect() {
    if (!prediction) return;
    try {
      await api.confirmPrediction(prediction.predictionId);
      setStatus('Marked as correct.');
      setConfirmed(true);
    } catch (err) {
      setStatus(`Could not confirm: ${err.message}`);
    }
  }

  return (
    <section className="grid two">
      <div className="card">
        <h2>Draw a digit</h2>
        <DigitCanvas onPixelsChange={handlePixelsChange} />
        <p className="muted">Pixel format sent to backend: 784 integers, 0 = background, 255 = ink.</p>
      </div>

      <div className="card">
        <h2>Prediction</h2>
        {pixels && <DigitPreview pixels={pixels} size={112} />}
        {loading && <p>Running kNN...</p>}
        {prediction ? (
          <div className="prediction-result">
            <div className="big-digit">{prediction.predictedLabel}</div>
            <p>Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
            <p>Latency: {prediction.responseTimeMs} ms</p>
            <p>k = {prediction.k}, method = {prediction.method}</p>
            <p>Dataset v{prediction.datasetVersion}, samples {prediction.sampleCount}</p>

            <div className="actions">
              {!confirmed && !feedbackSubmitted && (
                <button onClick={confirmCorrect}>Prediction is correct</button>
              )}
              {(confirmed || feedbackSubmitted) && (
                <button disabled>
                  {confirmed ? 'Marked as correct' : 'Feedback submitted'}
                </button>
              )}
            </div>

            {!(confirmed || feedbackSubmitted) && (
              <div className="feedback-box">
                <label>Wrong? Choose true label:</label>
                <select value={trueLabel} onChange={(e) => setTrueLabel(e.target.value)}>
                  <option value="">Select</option>
                  {[0,1,2,3,4,5,6,7,8,9].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
                <button disabled={trueLabel === ''} onClick={submitFeedback}>Submit feedback</button>
              </div>
            )}
          </div>
        ) : (
          <p>No prediction yet.</p>
        )}
        <p className="status">{status}</p>
      </div>
    </section>
  );
}
