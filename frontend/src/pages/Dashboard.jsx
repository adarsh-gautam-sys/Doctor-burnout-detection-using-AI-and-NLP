import { useState, useEffect, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Footer from '../components/Footer';
import { getDashboardData, exportCSV, exportJSON } from '../services/api';
import './Dashboard.css';

/* ═══════════════════════════════════════════════════════
   ANIMATED COUNTER
   ═══════════════════════════════════════════════════════ */
function Counter({ target, color }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const duration = 1200;
    const start = performance.now();
    const tick = (now) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      setCount(Math.round(target * (1 - Math.pow(1 - progress, 3))));
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [target]);

  return <span style={color ? { color } : undefined}>{count}</span>;
}

/* ═══════════════════════════════════════════════════════
   DETAIL MODAL
   ═══════════════════════════════════════════════════════ */
function DetailModal({ doctor, onClose }) {
  if (!doctor) return null;

  const riskColor = doctor.burnout === 'High' ? 'var(--red)'
    : doctor.burnout === 'Medium' ? 'var(--amber)' : 'var(--green)';

  return (
    <motion.div
      className="modal-overlay"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <motion.div
        className="modal glass-card"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        onClick={e => e.stopPropagation()}
      >
        <button className="modal-close" onClick={onClose}>✕</button>
        <h2>{doctor.name}</h2>
        <div className="modal-subtitle">
          {doctor.specialty || 'Unknown'} · ID: {doctor.id} · Last assessed: {doctor.last_updated || 'N/A'}
        </div>

        <div className="modal-stat-row">
          <div className="modal-stat">
            <div className="modal-stat-label">Risk Level</div>
            <div className="modal-stat-value" style={{ color: riskColor }}>{doctor.burnout || '—'}</div>
          </div>
          <div className="modal-stat">
            <div className="modal-stat-label">Confidence</div>
            <div className="modal-stat-value">{(doctor.confidence || 0).toFixed(1)}%</div>
          </div>
        </div>

        <div className="modal-section-title">Probability Breakdown</div>
        {[
          { label: 'Low', pct: doctor.low_pct || 0, color: 'var(--green)' },
          { label: 'Medium', pct: doctor.medium_pct || 0, color: 'var(--amber)' },
          { label: 'High', pct: doctor.high_pct || 0, color: 'var(--red)' },
        ].map(item => (
          <div className="modal-prob-row" key={item.label}>
            <span className="modal-prob-label">{item.label}</span>
            <div className="modal-prob-bar-bg">
              <motion.div
                className="modal-prob-bar"
                style={{ background: item.color }}
                initial={{ width: 0 }}
                animate={{ width: `${item.pct}%` }}
                transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
              />
            </div>
            <span className="modal-prob-pct">{item.pct.toFixed(1)}%</span>
          </div>
        ))}

        <div className="modal-section-title">Recommendation</div>
        <div className="modal-advice">
          <strong>For the physician: </strong>
          {doctor.doctor_advice || 'No specific recommendation available.'}
        </div>

        <div className="modal-section-title">Hospital Action</div>
        <div className="modal-advice" style={{ borderLeftColor: 'var(--blue)' }}>
          <strong>Hospital action: </strong>
          {doctor.hospital_advice || 'Standard monitoring protocol.'}
        </div>
      </motion.div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   DASHBOARD PAGE
   ═══════════════════════════════════════════════════════ */
export default function Dashboard() {
  const [allDoctors, setAllDoctors] = useState([]);
  const [hospitalStats, setHospitalStats] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filters
  const [search, setSearch] = useState('');
  const [deptFilter, setDeptFilter] = useState('');
  const [riskFilter, setRiskFilter] = useState('');

  // Sort
  const [sortField, setSortField] = useState('confidence');
  const [sortDir, setSortDir] = useState(-1);

  // Pagination
  const [displayCount, setDisplayCount] = useState(15);

  // Modal
  const [selectedDoctor, setSelectedDoctor] = useState(null);

  // Load data
  useEffect(() => {
    getDashboardData()
      .then(data => {
        setAllDoctors(data.doctors || []);
        setHospitalStats(data.hospital_stats || {});
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Departments
  const departments = useMemo(() =>
    [...new Set(allDoctors.map(d => d.specialty).filter(Boolean))].sort(),
    [allDoctors]
  );

  // Filtered & sorted
  const filteredDoctors = useMemo(() => {
    const searchLower = search.toLowerCase().trim();
    const riskOrder = { High: 3, Medium: 2, Low: 1 };

    let filtered = allDoctors.filter(d => {
      if (searchLower && !d.id.toLowerCase().includes(searchLower) &&
          !d.name.toLowerCase().includes(searchLower)) return false;
      if (deptFilter && d.specialty !== deptFilter) return false;
      if (riskFilter && d.burnout !== riskFilter) return false;
      return true;
    });

    filtered.sort((a, b) => {
      let valA, valB;
      if (sortField === 'burnout') {
        valA = riskOrder[a.burnout] || 0;
        valB = riskOrder[b.burnout] || 0;
      } else if (sortField === 'confidence') {
        valA = a.confidence || 0;
        valB = b.confidence || 0;
      } else {
        valA = (a[sortField] || '').toString().toLowerCase();
        valB = (b[sortField] || '').toString().toLowerCase();
      }
      if (valA < valB) return -1 * sortDir;
      if (valA > valB) return 1 * sortDir;
      return 0;
    });

    return filtered;
  }, [allDoctors, search, deptFilter, riskFilter, sortField, sortDir]);

  const handleSort = useCallback((field) => {
    if (sortField === field) {
      setSortDir(d => d * -1);
    } else {
      setSortField(field);
      setSortDir(-1);
    }
  }, [sortField]);

  useEffect(() => {
    setDisplayCount(15);
  }, [search, deptFilter, riskFilter]);

  const stats = hospitalStats;
  const total = stats.total_doctors || allDoctors.length || 1;

  const columns = [
    { key: 'id', label: 'ID' },
    { key: 'name', label: 'Name' },
    { key: 'specialty', label: 'Department' },
    { key: 'confidence', label: 'Confidence' },
    { key: 'burnout', label: 'Risk Level' },
    { key: 'last_updated', label: 'Last Updated' },
  ];

  return (
    <>
      <div className="dashboard">
        <div className="container">
          {/* Header */}
          <div className="page-header">
            <h1>Hospital Dashboard</h1>
            <p>Real-time physician burnout monitoring — data from <code>dashboard_data.json</code></p>
          </div>

          {/* Summary Cards */}
          <div className="summary-grid">
            <motion.div className="summary-card glass-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
              <div className="summary-card-label">Total Physicians</div>
              <div className="summary-card-value"><Counter target={stats.total_doctors || 0} /></div>
            </motion.div>
            <motion.div className="summary-card glass-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
              <div className="summary-card-label">High Risk</div>
              <div className="summary-card-value"><Counter target={stats.high_count || 0} color="var(--red)" /></div>
              <div className="summary-card-sub">{((stats.high_count || 0) / total * 100).toFixed(1)}% of physicians</div>
            </motion.div>
            <motion.div className="summary-card glass-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
              <div className="summary-card-label">Medium Risk</div>
              <div className="summary-card-value"><Counter target={stats.medium_count || 0} color="var(--amber)" /></div>
              <div className="summary-card-sub">{((stats.medium_count || 0) / total * 100).toFixed(1)}% of physicians</div>
            </motion.div>
            <motion.div className="summary-card glass-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
              <div className="summary-card-label">Low Risk</div>
              <div className="summary-card-value"><Counter target={stats.low_count || 0} color="var(--green)" /></div>
              <div className="summary-card-sub">{((stats.low_count || 0) / total * 100).toFixed(1)}% of physicians</div>
            </motion.div>
            <motion.div className="summary-card glass-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
              <div className="summary-card-label">Avg. Confidence</div>
              <div className="summary-card-value">
                {allDoctors.length > 0
                  ? (allDoctors.reduce((s, d) => s + (d.confidence || 0), 0) / allDoctors.length).toFixed(1) + '%'
                  : '—'}
              </div>
              <div className="summary-card-sub">Across all assessments</div>
            </motion.div>
          </div>

          {/* Distribution Chart */}
          <motion.div className="dist-section glass-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }}>
            <div className="dist-title">Burnout Risk Distribution</div>
            {[
              { label: 'Low', count: stats.low_count || 0, color: 'var(--green)' },
              { label: 'Medium', count: stats.medium_count || 0, color: 'var(--amber)' },
              { label: 'High', count: stats.high_count || 0, color: 'var(--red)' },
            ].map(item => (
              <div className="dist-row" key={item.label}>
                <span className="dist-label">{item.label}</span>
                <div className="dist-bar-bg">
                  <motion.div
                    className="dist-bar-fill"
                    style={{ background: item.color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${(item.count / total) * 100}%` }}
                    transition={{ duration: 1, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
                  >
                    {item.count > 0 && <span>{item.count}</span>}
                  </motion.div>
                </div>
                <span className="dist-count">{item.count}</span>
              </div>
            ))}
          </motion.div>

          {/* Filters */}
          <div className="filters-bar">
            <input
              type="text"
              className="filter-input"
              placeholder="Search by doctor ID or name..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            <select className="filter-select" value={deptFilter} onChange={e => setDeptFilter(e.target.value)}>
              <option value="">All Departments</option>
              {departments.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
            <select className="filter-select" value={riskFilter} onChange={e => setRiskFilter(e.target.value)}>
              <option value="">All Risk Levels</option>
              <option value="High">High</option>
              <option value="Medium">Medium</option>
              <option value="Low">Low</option>
            </select>
            <button className="export-btn" onClick={() => exportCSV(filteredDoctors)}>⬇ Export CSV</button>
            <button className="export-btn export-btn--outline" onClick={() => exportJSON(filteredDoctors, hospitalStats)}>⬇ JSON</button>
          </div>

          {/* Loading / Error */}
          {loading && <div className="table-status">Loading dashboard data...</div>}
          {error && <div className="table-status table-status--error">Error: {error}</div>}

          {/* Doctor Table */}
          {!loading && !error && (
            <motion.div className="table-container" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}>
              <table className="doc-table">
                <thead>
                  <tr>
                    {columns.map(col => (
                      <th
                        key={col.key}
                        className={sortField === col.key ? 'active' : ''}
                        onClick={() => handleSort(col.key)}
                      >
                        {col.label}
                        <span className="sort-icon">
                          {sortField === col.key ? (sortDir === 1 ? '▲' : '▼') : '▲'}
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredDoctors.slice(0, displayCount).map((doc, i) => (
                    <motion.tr
                      key={doc.id}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: Math.min(i * 0.02, 0.5) }}
                      onClick={() => setSelectedDoctor(doc)}
                      className="doc-row"
                    >
                      <td className="mono-cell">{doc.id}</td>
                      <td>{doc.name}</td>
                      <td className="muted-cell">{doc.specialty || '—'}</td>
                      <td className="mono-cell">{doc.confidence ? doc.confidence.toFixed(1) + '%' : '—'}</td>
                      <td>
                        <span className={`risk-badge ${(doc.burnout || '').toLowerCase()}`}>
                          {doc.burnout || '—'}
                        </span>
                      </td>
                      <td className="mono-cell muted-cell">{doc.last_updated || '—'}</td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>

              <div className="table-footer">
                <span className="table-info">
                  Showing {Math.min(displayCount, filteredDoctors.length)} of {filteredDoctors.length} physicians
                </span>
                {filteredDoctors.length > displayCount && (
                  <button className="load-more-btn" onClick={() => setDisplayCount(filteredDoctors.length)}>
                    Load All ({filteredDoctors.length - displayCount} more)
                  </button>
                )}
              </div>
            </motion.div>
          )}
        </div>
      </div>

      {/* Detail Modal */}
      <AnimatePresence>
        {selectedDoctor && (
          <DetailModal doctor={selectedDoctor} onClose={() => setSelectedDoctor(null)} />
        )}
      </AnimatePresence>

      <Footer />
    </>
  );
}
