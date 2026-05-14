const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function getJson(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`Request failed for ${path}`);
  }
  return response.json();
}

export function fetchHealth() {
  return getJson('/health');
}

export function fetchMetrics() {
  return getJson('/metrics?limit=120');
}

export function fetchAnomalies() {
  return getJson('/anomalies?limit=40');
}

export function fetchModels() {
  return getJson('/models');
}

export function fetchForecast(contextWindow) {
  if (!contextWindow.length) {
    return Promise.resolve({ forecast: [] });
  }
  return getJson('/forecast', {
    method: 'POST',
    body: JSON.stringify({
      sensor_id: contextWindow.at(-1).sensor_id,
      target: 'temperature_k',
      horizon: 12,
      context: contextWindow,
    }),
  });
}
