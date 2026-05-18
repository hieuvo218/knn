import React, { useEffect, useState } from 'react';
import { api } from '../api.js';
import DigitPreview from '../components/DigitPreview.jsx';

export default function Database() {
  const [stats, setStats] = useState(null);
  const [samples, setSamples] = useState(null);
  const [idFilter, setIdFilter] = useState('');
  const [sourceFilter, setSourceFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [orderBy, setOrderBy] = useState('latest');
  const [page, setPage] = useState(0);
  const [message, setMessage] = useState('');

  async function load(nextPage = page) {
    const [s, rows] = await Promise.all([
      api.dbStats(),
      api.dbSamples({ page: nextPage, size: 20, id: idFilter, status: statusFilter, source: sourceFilter, order: orderBy }),
    ]);
    setStats(s);
    setSamples(rows);
    setPage(nextPage);
  }

  useEffect(() => { load(0).catch(err => setMessage(err.message)); }, []);

  async function search() {
    await load(0);
  }

  async function updateLabel(id) {
    const value = prompt('New true label 0-9:');
    if (value === null) return;
    await api.updateSample(id, Number(value));
    setMessage(`Updated sample ${id}`);
    await load();
  }

  async function deleteSample(id) {
    if (!confirm(`Delete sample ${id}?`)) return;
    await api.deleteSample(id);
    setMessage(`Deleted sample ${id}`);
    await load();
  }

  return (
    <div className="stack">
      <section className="card">
        <h2>Database Stats</h2>
        {stats ? (
          <>
            <div className="metrics">
              <Metric label="Total" value={stats.totalSamples} />
              <Metric label="Accepted" value={stats.acceptedSamples} />
              <Metric label="Pending feedback" value={stats.pendingFeedback} />
              <Metric label="Dataset version" value={stats.datasetVersion} />
            </div>
            <h3>Distribution</h3>
            <div className="bars">
              {stats.distribution.map(row => <div key={row.label}><span>{row.label}</span><div><i style={{ width: `${Math.min(100, row.count / Math.max(1, stats.acceptedSamples) * 1000)}%` }} /></div><b>{row.count}</b></div>)}
            </div>
          </>
        ) : <p>Loading...</p>}
      </section>

      <section className="card">
        <h2>Query Samples</h2>
        <div className="form-row">
          <label>ID</label>
          <input type="number" value={idFilter} onChange={e => setIdFilter(e.target.value)} placeholder="e.g. 12" />
          <label>Status</label>
          <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
            <option value="">all</option>
            <option value="accepted">accepted</option>
            <option value="pending">pending</option>
            <option value="rejected">rejected</option>
          </select>
          <label>Source</label>
          <select value={sourceFilter} onChange={e => setSourceFilter(e.target.value)}>
            <option value="">all</option>
            <option value="mnist_npz">mnist_npz</option>
            <option value="mnist_csv">mnist_csv</option>
            <option value="feedback">feedback</option>
            <option value="feedback_submission">feedback_submission</option>
          </select>
          <label>Order</label>
          <select value={orderBy} onChange={e => setOrderBy(e.target.value)}>
            <option value="latest">latest</option>
            <option value="newest">newest</option>
            <option value="oldest">oldest</option>
            <option value="id_desc">id desc</option>
            <option value="id_asc">id asc</option>
          </select>
          <button onClick={search}>Search</button>
        </div>

        {samples && <p>Total matched: {samples.total}</p>}
        {samples && (
          <div className="sample-grid">
            {samples.rows.map(row => (
              <div className="sample-card" key={row.id}>
                <DigitPreview pixels={row.pixels} size={84} />
                <p>ID #{row.id}</p>
                <p>Label: {row.label}</p>
                <p>{row.source} · {row.status} · {row.rowType}</p>
                {row.rowType === 'digit' && <button onClick={() => updateLabel(row.id)}>Edit label</button>}
                {row.rowType === 'digit' && <button className="danger" onClick={() => deleteSample(row.id)}>Delete</button>}
              </div>
            ))}
          </div>
        )}
        <div className="pager">
          <button disabled={page <= 0} onClick={() => load(page - 1)}>Prev</button>
          <span>Page {page}</span>
          <button disabled={!samples || (page + 1) * samples.size >= samples.total} onClick={() => load(page + 1)}>Next</button>
        </div>
      </section>
      <p className="status">{message}</p>
    </div>
  );
}

function Metric({ label, value }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}
