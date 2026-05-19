import React, { useEffect, useState } from 'react';
import { api } from '../api.js';
import DigitPreview from '../components/DigitPreview.jsx';

export default function Model() {
  const [dashboard, setDashboard] = useState(null);
  const [feedback, setFeedback] = useState([]);
  const [sampleCount, setSampleCount] = useState(500);
  const [method, setMethod] = useState('kd_tree');
  const [kText, setKText] = useState('1,3,5,7');
  const [tuneResult, setTuneResult] = useState(null);
  const [tuneMetric, setTuneMetric] = useState('accuracy');
  const [status, setStatus] = useState('');

  async function load() {
    const [dash, fb] = await Promise.all([api.dashboard(), api.listFeedback('pending')]);
    setDashboard(dash);
    setFeedback(fb);
  }

  useEffect(() => { load().catch(err => setStatus(err.message)); }, []);

  async function runTune() {
    setStatus('Tuning...');
    try {
      const kValues = kText.split(',').map(v => Number(v.trim())).filter(v => Number.isInteger(v) && v > 0);
      const result = await api.tune({ sampleCount: Number(sampleCount), method, kValues });
      setTuneResult(result);
      setStatus('Tune done.');
      await load();
    } catch (err) {
      setStatus(`Tune failed: ${err.message}`);
    }
  }

  async function activate(row) {
    try {
      await api.activateTune({ jobId: tuneResult.jobId, k: row.k, method: row.method });
      setStatus(`Activated k=${row.k}, method=${row.method}`);
      await load();
    } catch (err) {
      setStatus(`Activate failed: ${err.message}`);
    }
  }

  async function accept(id) {
    await api.acceptFeedback(id);
    await load();
  }

  async function reject(id) {
    await api.rejectFeedback(id);
    await load();
  }

  const grouped = feedback.reduce((acc, item) => {
    const key = item.trueLabel;
    acc[key] = acc[key] || [];
    acc[key].push(item);
    return acc;
  }, {});

  return (
    <div className="stack">
      <section className="card">
        <h2>Model Dashboard</h2>
        {dashboard ? (
          <div className="metrics">
            <Metric label="Predictions" value={dashboard.totalPredictions} />
            <Metric label="Accuracy" value={`${(dashboard.accuracy * 100).toFixed(1)}%`} />
            <Metric label="F1-score" value={dashboard.f1Score.toFixed(3)} />
            <Metric label="Avg response" value={`${dashboard.avgResponseTimeMs.toFixed(1)} ms`} />
            <Metric label="Accepted samples" value={dashboard.acceptedSamples} />
            <Metric label="Pending feedback" value={dashboard.pendingFeedback} />
          </div>
        ) : <p>Loading...</p>}
        {dashboard && <p>Active: k={dashboard.activeModel.k}, method={dashboard.activeModel.method}, dataset v{dashboard.datasetVersion}</p>}
      </section>

      <section className="card">
        <h2>Confusion Matrix</h2>
        {dashboard && <ConfusionMatrix matrix={dashboard.confusionMatrix} />}
      </section>

      <section className="card">
        <h2>Tuning</h2>
        <div className="form-row">
          <label>Sample count</label>
          <input type="number" value={sampleCount} onChange={e => setSampleCount(e.target.value)} />
          <label>Method</label>
          <select value={method} onChange={e => setMethod(e.target.value)}>
            <option value="kd_tree">kd-tree</option>
            <option value="lsh">LSH</option>
          </select>
          <label>k values</label>
          <input value={kText} onChange={e => setKText(e.target.value)} />
          <button onClick={runTune}>Run tune</button>
        </div>
        {tuneResult && (
          <>
          <div className="tune-plot-controls" style={{marginBottom:10}}>
            <label>View metric:</label>
            <select value={tuneMetric} onChange={e => setTuneMetric(e.target.value)}>
              <option value="accuracy">Accuracy</option>
              <option value="f1Score">F1</option>
              <option value="avgLatencyMs">Latency (ms)</option>
            </select>
          </div>
          <table>
            <thead><tr><th>Rank</th><th>k</th><th>Method</th><th>Accuracy</th><th>F1</th><th>Latency</th><th>Train</th><th>Eval</th><th>Action</th></tr></thead>
            <tbody>
              {tuneResult.topResults.map(row => (
                <tr key={`${row.k}-${row.method}`}>
                  <td>{row.rank}</td><td>{row.k}</td><td>{row.method}</td>
                  <td>{(row.accuracy * 100).toFixed(2)}%</td>
                  <td>{row.f1Score.toFixed(3)}</td>
                  <td>{row.avgLatencyMs.toFixed(2)} ms</td>
                  <td>{row.trainingSamples}</td>
                  <td>{row.evaluatedSamples}</td>
                  <td><button onClick={() => activate(row)}>Use this k</button></td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{marginTop:12}}>
            <TuneChart results={tuneResult.topResults} metric={tuneMetric} />
          </div>
          </>
        )}
      </section>

      <section className="card">
        <h2>Pending Feedback</h2>
        {Object.keys(grouped).length === 0 && <p>No pending feedback.</p>}
        {Object.entries(grouped).map(([label, items]) => (
          <div key={label} className="feedback-group">
            <h3>True label: {label}</h3>
            <div className="feedback-grid">
              {items.map(item => (
                <div className="feedback-card" key={item.id}>
                  <DigitPreview pixels={item.pixels} size={84} />
                  <p>ID #{item.id}</p>
                  <p>Pred: {item.predictedLabel ?? 'N/A'}</p>
                  <button onClick={() => accept(item.id)}>Accept</button>
                  <button className="danger" onClick={() => reject(item.id)}>Reject</button>
                </div>
              ))}
            </div>
          </div>
        ))}
      </section>
      <p className="status">{status}</p>
    </div>
  );
}

function Metric({ label, value }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function ConfusionMatrix({ matrix }) {
  return (
    <table className="matrix">
      <thead><tr><th>true\\pred</th>{[0,1,2,3,4,5,6,7,8,9].map(n => <th key={n}>{n}</th>)}</tr></thead>
      <tbody>
        {matrix.map((row, i) => <tr key={i}><th>{i}</th>{row.map((v, j) => <td key={j}>{v}</td>)}</tr>)}
      </tbody>
    </table>
  );
}

function TuneChart({ results, metric }) {
  if (!results || results.length === 0) return <p>No data to plot.</p>;

  // sort by k
  const rows = [...results].sort((a,b) => a.k - b.k);
  const values = rows.map(r => r[metric]);
  const ks = rows.map(r => r.k);

  const width = 480;
  const height = 180;
  const pad = 30;

  const vmin = Math.min(...values);
  const vmax = Math.max(...values);
  const yrange = vmax - vmin || 1;

  const points = values.map((v, i) => {
    const x = pad + (i / Math.max(1, values.length - 1)) * (width - pad * 2);
    const y = pad + (1 - (v - vmin) / yrange) * (height - pad * 2);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div>
      <svg width={width} height={height} style={{border:'1px solid #eee', background:'#fff'}}>
        {/* grid lines */}
        {[0,0.25,0.5,0.75,1].map((g,i) => {
          const y = pad + g * (height - pad * 2);
          return <line key={i} x1={pad} x2={width-pad} y1={y} y2={y} stroke="#f0f0f0" />
        })}

        {/* polyline */}
        <polyline fill="none" stroke="#2563eb" strokeWidth="2" points={points} />

        {/* points */}
        {values.map((v,i) => {
          const coords = points.split(' ')[i].split(',');
          return <circle key={i} cx={coords[0]} cy={coords[1]} r={4} fill="#fff" stroke="#2563eb" />;
        })}

        {/* x labels */}
        {ks.map((k,i) => {
          const x = pad + (i / Math.max(1, ks.length - 1)) * (width - pad * 2);
          return <text key={i} x={x} y={height - 6} fontSize={10} textAnchor="middle">{k}</text>;
        })}

        {/* y labels */}
        {[0,0.25,0.5,0.75,1].map((g,i) => {
          const v = vmin + (1 - g) * yrange;
          const y = pad + g * (height - pad * 2);
          const label = metric === 'avgLatencyMs' ? `${v.toFixed(1)} ms` : (metric === 'accuracy' || metric === 'f1Score' ? `${(v*100).toFixed(1)}%` : v.toFixed(3));
          return <text key={i} x={6} y={y+4} fontSize={10}>{label}</text>;
        })}
      </svg>
      <div style={{fontSize:12, color:'#444', marginTop:6}}>
        <strong>Metric:</strong> {metric} &nbsp; • &nbsp; <strong>k:</strong> {ks.join(', ')}
      </div>
    </div>
  );
}
