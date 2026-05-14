import { create } from 'zustand';

function isFiniteNumber(value) {
  return Number.isFinite(Number(value));
}

function normalizeMetric(metric) {
  if (!metric || typeof metric !== 'object') {
    return null;
  }
  if (!metric.timestamp || !metric.sensor_id) {
    return null;
  }
  if (!isFiniteNumber(metric.temperature_k) || !isFiniteNumber(metric.pressure_atm) || !isFiniteNumber(metric.vibration_mms)) {
    return null;
  }
  return metric;
}

function normalizeAnomaly(anomaly) {
  if (!anomaly || typeof anomaly !== 'object') {
    return null;
  }
  if (!anomaly.timestamp || !anomaly.sensor_id || !anomaly.model_name) {
    return null;
  }
  if (!isFiniteNumber(anomaly.anomaly_score)) {
    return null;
  }
  return anomaly;
}

const useStore = create((set) => ({
  activePage: 'dashboard',
  liveMetrics: [],
  anomalies: [],
  forecast: [],
  catalog: [],
  health: null,
  apiStatus: {
    health: 'idle',
    metrics: 'idle',
    anomalies: 'idle',
    models: 'idle',
    forecast: 'idle',
  },
  isConnected: false,

  setActivePage: (activePage) => set({ activePage }),
  setConnected: (isConnected) => set({ isConnected }),
  setHealth: (health) => set({ health }),
  setCatalog: (catalog) => set({ catalog }),
  setForecast: (forecast) => set({ forecast }),
  setApiStatus: (key, value) =>
    set((state) => ({
      apiStatus: {
        ...state.apiStatus,
        [key]: value,
      },
    })),
  setInitialMetrics: (metrics) =>
    set({
      liveMetrics: (Array.isArray(metrics) ? metrics : []).map(normalizeMetric).filter(Boolean),
    }),
  setInitialAnomalies: (anomalies) =>
    set({
      anomalies: (Array.isArray(anomalies) ? anomalies : []).map(normalizeAnomaly).filter(Boolean),
    }),
  addMetric: (metric) =>
    set((state) => {
      const item = normalizeMetric(metric);
      if (!item) {
        return state;
      }
      return { liveMetrics: [...state.liveMetrics, item].slice(-180) };
    }),
  addMetricsBatch: (metrics) =>
    set((state) => {
      const batch = (Array.isArray(metrics) ? metrics : []).map(normalizeMetric).filter(Boolean);
      if (!batch.length) {
        return state;
      }
      return { liveMetrics: [...state.liveMetrics, ...batch].slice(-180) };
    }),
  addAnomaly: (anomaly) =>
    set((state) => {
      const item = normalizeAnomaly(anomaly);
      if (!item) {
        return state;
      }
      return { anomalies: [item, ...state.anomalies].slice(0, 80) };
    }),
  addAnomaliesBatch: (anomalies) =>
    set((state) => {
      const batch = (Array.isArray(anomalies) ? anomalies : []).map(normalizeAnomaly).filter(Boolean);
      if (!batch.length) {
        return state;
      }
      return { anomalies: [...batch, ...state.anomalies].slice(0, 80) };
    }),
}));

export default useStore;
