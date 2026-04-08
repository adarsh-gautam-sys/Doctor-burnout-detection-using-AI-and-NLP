const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Upload an image/text file and get burnout prediction + XAI explanation.
 * @param {File} file 
 * @returns {Promise<Object>} prediction result
 */
export async function uploadAndPredict(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/ocr`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Prediction failed' }));
    throw new Error(err.detail || 'Prediction failed');
  }

  return res.json();
}

/**
 * Fetch dashboard data from the static JSON file.
 * @returns {Promise<Object>} { hospital_stats, doctors }
 */
export async function getDashboardData() {
  const res = await fetch(`${API_BASE}/data/dashboard_data.json?t=${Date.now()}`);
  if (!res.ok) throw new Error('Failed to load dashboard data');
  return res.json();
}

/**
 * Fetch dataset summary info.
 * @returns {Promise<Object>} { summary: { total_images, ... } }
 */
export async function getDatasetData() {
  const res = await fetch(`${API_BASE}/data/dataset.json`);
  if (!res.ok) throw new Error('Failed to load dataset data');
  return res.json();
}

/**
 * Fetch model results.
 * @returns {Promise<Object>}
 */
export async function getModelResults() {
  const res = await fetch(`${API_BASE}/results`);
  if (!res.ok) throw new Error('Failed to load results');
  return res.json();
}

/**
 * Download a CSV report from the backend.
 */
export async function downloadReport() {
  const res = await fetch(`${API_BASE}/generate_report`);
  if (!res.ok) throw new Error('Failed to generate report');
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `burnout_report_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Export filtered doctors list as CSV (client-side).
 * @param {Array} doctors 
 */
export function exportCSV(doctors) {
  const recommendations = {
    High: 'Immediate intervention required. Reduce workload and arrange support.',
    Medium: 'Monitor closely. Schedule follow-up assessment within 2 weeks.',
    Low: 'No immediate action needed. Continue routine monitoring.',
  };

  const headers = [
    'Doctor ID', 'Name', 'Specialty', 'Burnout Risk', 'Confidence (%)',
    'Low %', 'Medium %', 'High %', 'Last Updated', 'Recommendation',
  ];

  const rows = doctors.map(d => [
    d.id, d.name, d.specialty || '', d.burnout || '',
    (d.confidence || 0).toFixed(1),
    (d.low_pct || 0).toFixed(1),
    (d.medium_pct || 0).toFixed(1),
    (d.high_pct || 0).toFixed(1),
    d.last_updated || '',
    recommendations[d.burnout] || 'Assessment pending',
  ]);

  let csv = '\uFEFF' + headers.join(',') + '\n';
  rows.forEach(row => {
    csv += row.map(v => `"${String(v).replace(/"/g, '""')}"`).join(',') + '\n';
  });

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `burnout_report_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Export filtered doctors list as JSON (client-side).
 * @param {Array} doctors 
 * @param {Object} hospitalStats 
 */
export function exportJSON(doctors, hospitalStats) {
  const data = {
    exported_at: new Date().toISOString(),
    total_doctors: doctors.length,
    hospital_stats: hospitalStats,
    doctors,
  };

  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `burnout_data_${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}
