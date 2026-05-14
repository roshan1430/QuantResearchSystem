import { Suspense, lazy, useEffect, useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Cpu,
  Database,
  Orbit,
  Radio,
  ShieldAlert,
} from 'lucide-react';
import usePlatformData from './hooks/usePlatformData';
import wsService from './services/websocket';
import useStore from './store/useStore';
import AppErrorBoundary from './components/AppErrorBoundary';

const queryClient = new QueryClient();
const TimeSeriesChart = lazy(() => import('./components/TimeSeriesChart'));

function formatFixed(value, digits = 3) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : '0.000';
}

function getForecastModelSummary(catalog) {
  const trainedModel = catalog.find((item) => item.task_type === 'forecasting' && item.status === 'trained');
  if (trainedModel) {
    return { name: trainedModel.model_name, status: trainedModel.status, accent: 'text-emerald-300', chip: 'border-emerald-400/25 bg-emerald-500/10 text-emerald-200' };
  }
  const fallbackModel = catalog.find((item) => item.task_type === 'forecasting');
  if (fallbackModel) {
    return {
      name: fallbackModel.model_name,
      status: fallbackModel.status,
      accent: 'text-amber-300',
      chip: 'border-amber-400/25 bg-amber-500/10 text-amber-200',
    };
  }
  return { name: 'unavailable', status: 'unknown', accent: 'text-slate-400', chip: 'border-slate-700 bg-slate-900/70 text-slate-300' };
}

const pages = [
  { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
  { id: 'stream', label: 'Live Stream Monitor', icon: Radio },
  { id: 'anomaly', label: 'Anomaly Detection', icon: ShieldAlert },
  { id: 'forecast', label: 'Forecasting', icon: Orbit },
  { id: 'metrics', label: 'Model Metrics', icon: Activity },
  { id: 'health', label: 'System Health', icon: Database },
];

function Sidebar() {
  const { activePage, setActivePage } = useStore();

  return (
    <aside className="w-full border-b border-slate-800/80 bg-slate-950/90 lg:w-72 lg:border-b-0 lg:border-r">
      <div className="flex items-center gap-3 border-b border-slate-800/80 px-5 py-5">
        <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-cyan-400/15 text-cyan-300 ring-1 ring-cyan-400/20">
          <Cpu size={22} />
        </div>
        <div>
          <p className="font-display text-lg text-slate-50">RT Time-Series Lab</p>
          <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Scientific ML Platform</p>
        </div>
      </div>
      <nav className="grid grid-cols-2 gap-2 p-4 lg:grid-cols-1">
        {pages.map((page) => {
          const Icon = page.icon;
          const isActive = activePage === page.id;
          return (
            <button
              key={page.id}
              type="button"
              onClick={() => setActivePage(page.id)}
              className={`flex items-center gap-3 rounded-2xl border px-4 py-3 text-left transition ${
                isActive
                  ? 'border-cyan-400/40 bg-cyan-400/10 text-cyan-200'
                  : 'border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700 hover:text-slate-100'
              }`}
            >
              <Icon size={18} />
              <span className="text-sm font-medium">{page.label}</span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
}

function PageShell({ title, subtitle, children }) {
  return (
    <section className="flex flex-col gap-5">
      <header className="rounded-3xl border border-slate-800 bg-[radial-gradient(circle_at_top_left,_rgba(34,211,238,0.14),_transparent_28%),linear-gradient(135deg,_rgba(15,23,42,0.96),_rgba(2,6,23,0.94))] p-6 shadow-[0_22px_60px_rgba(2,6,23,0.45)]">
        <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/85">Research Console</p>
        <h1 className="mt-2 font-display text-3xl text-slate-50">{title}</h1>
        <p className="mt-2 max-w-3xl text-sm text-slate-400">{subtitle}</p>
      </header>
      {children}
    </section>
  );
}

function StatCard({ label, value, accent }) {
  return (
    <div className="rounded-3xl border border-slate-800/90 bg-slate-950/60 p-5 shadow-[0_18px_45px_rgba(2,6,23,0.35)]">
      <p className="text-xs uppercase tracking-[0.25em] text-slate-400">{label}</p>
      <p className={`mt-3 font-mono text-3xl ${accent}`}>{value}</p>
    </div>
  );
}

function DashboardPage() {
  const { liveMetrics, anomalies, forecast, isConnected, catalog } = useStore();
  const latest = liveMetrics.at(-1);
  const forecastModel = getForecastModelSummary(catalog);

  return (
    <PageShell
      title="Dashboard"
      subtitle="Unified experimental view across streaming telemetry, anomalies, forecasts, and platform state."
    >
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Temperature" value={`${formatFixed(latest?.temperature_k, 3)} K`} accent="text-cyan-300" />
        <StatCard label="Pressure" value={`${formatFixed(latest?.pressure_atm, 3)} atm`} accent="text-emerald-300" />
        <StatCard label="Vibration" value={`${formatFixed(latest?.vibration_mms, 3)} mm/s`} accent="text-amber-300" />
        <StatCard label="Link State" value={isConnected ? 'Live' : 'Offline'} accent={isConnected ? 'text-rose-300' : 'text-slate-500'} />
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.8fr,1fr]">
        <div className="panel">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="panel-label">Multi-Signal Timeline</p>
              <h2 className="panel-title">Streaming Measurements</h2>
            </div>
          </div>
          <Suspense fallback={<div className="h-[420px] animate-pulse rounded-2xl bg-slate-900/60" />}>
            <TimeSeriesChart />
          </Suspense>
        </div>
        <div className="panel">
          <p className="panel-label">Anomaly Feed</p>
          <h2 className="panel-title">Latest Alerts</h2>
          <div className="mt-4 space-y-3">
            {anomalies.slice(0, 6).map((item) => (
              <div key={`${item.timestamp}-${item.model_name}`} className="rounded-2xl border border-rose-400/20 bg-rose-500/5 p-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase tracking-[0.2em] text-rose-300">{item.model_name}</span>
                  <span className="font-mono text-xs text-slate-500">{new Date(item.timestamp).toLocaleTimeString()}</span>
                </div>
                <p className="mt-2 text-sm text-slate-100">{item.sensor_id}</p>
                <p className="mt-1 text-xs text-slate-400">Score {formatFixed(item.anomaly_score, 4)}</p>
              </div>
            ))}
            {anomalies.length === 0 && <p className="text-sm text-slate-500">No anomaly events in the current buffer.</p>}
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="panel-label">Forecast Summary</p>
            <h2 className="panel-title">Current Horizon Projection</h2>
          </div>
          <div className={`rounded-full border px-3 py-1 text-[11px] uppercase tracking-[0.22em] ${forecastModel.chip}`}>
            {forecastModel.status}
          </div>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-4">
          {forecast.slice(0, 4).map((point) => (
            <div key={point.step} className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Step {point.step}</p>
              <p className="mt-2 font-mono text-xl text-cyan-200">{formatFixed(point.value, 3)} K</p>
            </div>
          ))}
          {forecast.length === 0 && <p className="text-sm text-slate-500">Forecast context will appear after the first API refresh.</p>}
        </div>
        <p className={`mt-4 text-xs uppercase tracking-[0.18em] ${forecastModel.accent}`}>Active forecast model: {forecastModel.name}</p>
      </div>
    </PageShell>
  );
}

function StreamPage() {
  const { liveMetrics } = useStore();
  return (
    <PageShell
      title="Live Stream Monitor"
      subtitle="Inspect the incoming streaming window, engineered features, and temporal response under load."
    >
      <div className="panel">
        <Suspense fallback={<div className="h-[420px] animate-pulse rounded-2xl bg-slate-900/60" />}>
          <TimeSeriesChart />
        </Suspense>
      </div>
      <div className="panel overflow-x-auto">
        <p className="panel-label">Latest Feature Rows</p>
        <h2 className="panel-title">Engineered Sensor Window</h2>
        <table className="mt-4 min-w-full text-sm">
          <thead className="text-slate-500">
            <tr>
              <th className="px-3 py-2 text-left">Timestamp</th>
              <th className="px-3 py-2 text-left">Temp</th>
              <th className="px-3 py-2 text-left">Z-score</th>
              <th className="px-3 py-2 text-left">Momentum</th>
              <th className="px-3 py-2 text-left">Volatility</th>
            </tr>
          </thead>
          <tbody>
            {liveMetrics.slice(-12).reverse().map((metric) => (
              <tr key={metric.timestamp} className="border-t border-slate-800 text-slate-300">
                <td className="px-3 py-2 font-mono text-xs">{new Date(metric.timestamp).toLocaleTimeString()}</td>
                <td className="px-3 py-2">{formatFixed(metric.temperature_k, 4)}</td>
                <td className="px-3 py-2">{Number.isFinite(Number(metric.z_score_16)) ? formatFixed(metric.z_score_16, 4) : 'n/a'}</td>
                <td className="px-3 py-2">{Number.isFinite(Number(metric.momentum_8)) ? formatFixed(metric.momentum_8, 4) : 'n/a'}</td>
                <td className="px-3 py-2">{Number.isFinite(Number(metric.volatility_16)) ? formatFixed(metric.volatility_16, 4) : 'n/a'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </PageShell>
  );
}

function AnomalyPage() {
  const { anomalies } = useStore();
  return (
    <PageShell
      title="Anomaly Detection"
      subtitle="Track model outputs, thresholds, and first-order explanations from the anomaly pipeline."
    >
      <div className="grid gap-5 lg:grid-cols-2">
        {anomalies.slice(0, 10).map((item) => (
          <div key={`${item.timestamp}-${item.model_name}`} className="panel">
            <div className="flex items-center justify-between">
              <div>
                <p className="panel-label">{item.model_name}</p>
                <h2 className="panel-title">{item.sensor_id}</h2>
              </div>
              <AlertTriangle className="text-rose-300" />
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              <StatCard label="Score" value={formatFixed(item.anomaly_score, 4)} accent="text-rose-300" />
              <StatCard label="Threshold" value={Number.isFinite(Number(item.threshold)) ? formatFixed(item.threshold, 4) : 'n/a'} accent="text-amber-300" />
              <StatCard label="State" value={item.is_anomaly ? 'Anomaly' : 'Normal'} accent="text-cyan-300" />
            </div>
          </div>
        ))}
        {anomalies.length === 0 && <div className="panel text-slate-500">Awaiting anomaly events from the ML engine.</div>}
      </div>
    </PageShell>
  );
}

function ForecastPage() {
  const { forecast, catalog } = useStore();
  const forecastModel = getForecastModelSummary(catalog);
  return (
    <PageShell
      title="Forecasting"
      subtitle="Short-horizon forecasts generated from the model-serving engine for controlled extrapolation studies."
    >
      <div className="panel">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="panel-label">Forecast Horizon</p>
            <h2 className="panel-title">Projected Temperature Track</h2>
          </div>
          <div className={`rounded-full border px-3 py-1 text-[11px] uppercase tracking-[0.22em] ${forecastModel.chip}`}>
            {forecastModel.status}
          </div>
        </div>
        <p className={`mt-3 text-xs uppercase tracking-[0.18em] ${forecastModel.accent}`}>Active model: {forecastModel.name}</p>
        <div className="mt-5 grid gap-3 md:grid-cols-3 xl:grid-cols-6">
          {forecast.map((point) => (
            <div key={point.step} className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">t + {point.step}</p>
              <p className="mt-2 font-mono text-xl text-cyan-200">{formatFixed(point.value, 3)}</p>
            </div>
          ))}
        </div>
      </div>
    </PageShell>
  );
}

function MetricsPage() {
  const { catalog } = useStore();
  return (
    <PageShell
      title="Model Metrics"
      subtitle="Serving inventory for online, trained, and roadmap models spanning anomaly detection, regression, and forecasting."
    >
      <div className="panel overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="text-slate-500">
            <tr>
              <th className="px-3 py-2 text-left">Model</th>
              <th className="px-3 py-2 text-left">Task</th>
              <th className="px-3 py-2 text-left">Framework</th>
              <th className="px-3 py-2 text-left">Status</th>
            </tr>
          </thead>
          <tbody>
            {catalog.map((item) => (
              <tr key={item.model_name} className="border-t border-slate-800 text-slate-300">
                <td className="px-3 py-2">{item.model_name}</td>
                <td className="px-3 py-2">{item.task_type}</td>
                <td className="px-3 py-2">{item.framework}</td>
                <td className="px-3 py-2">
                  <span
                    className={`rounded-full border px-2 py-1 text-[11px] uppercase tracking-[0.18em] ${
                      item.status === 'trained'
                        ? 'border-emerald-400/25 bg-emerald-500/10 text-emerald-200'
                        : item.status.includes('baseline')
                          ? 'border-amber-400/25 bg-amber-500/10 text-amber-200'
                          : item.status === 'online'
                            ? 'border-cyan-400/25 bg-cyan-500/10 text-cyan-200'
                            : 'border-slate-700 bg-slate-900/70 text-slate-300'
                    }`}
                  >
                    {item.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </PageShell>
  );
}

function HealthPage() {
  const { health } = useStore();
  return (
    <PageShell
      title="System Health"
      subtitle="Operational visibility across the API gateway, database, Kafka layer, and model-serving interface."
    >
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Status" value={health?.status ?? 'unknown'} accent="text-cyan-300" />
        <StatCard label="Database" value={health?.database ?? 'unknown'} accent="text-emerald-300" />
        <StatCard label="Kafka" value={health?.kafka ?? 'unknown'} accent="text-amber-300" />
        <StatCard label="ML Engine" value={health?.ml_engine ?? 'unknown'} accent="text-rose-300" />
      </div>
    </PageShell>
  );
}

function MainView() {
  const { activePage } = useStore();
  usePlatformData();

  const views = {
    dashboard: <DashboardPage />,
    stream: <StreamPage />,
    anomaly: <AnomalyPage />,
    forecast: <ForecastPage />,
    metrics: <MetricsPage />,
    health: <HealthPage />,
  };

  return <div className="flex-1 overflow-y-auto p-4 lg:p-8">{views[activePage]}</div>;
}

function AppShell() {
  const [bootstrapped, setBootstrapped] = useState(false);
  const { apiStatus } = useStore();
  const hasApiError = Object.values(apiStatus).some((status) => status === 'error');

  useEffect(() => {
    const url = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/live';
    wsService.connect(url);
    setBootstrapped(true);
    return () => wsService.disconnect();
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground">
      {hasApiError && (
        <div className="border-b border-amber-500/40 bg-amber-500/15 px-4 py-2 text-center text-xs text-amber-200">
          API connectivity issue detected. Check backend (`:8000`) and ML engine (`:8001`) processes.
        </div>
      )}
      <div className="mx-auto flex min-h-screen max-w-[1800px] flex-col lg:flex-row">
        <Sidebar />
        <MainView />
      </div>
      {!bootstrapped && <div className="fixed bottom-4 right-4 rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-xs text-slate-400">Bootstrapping dashboard</div>}
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppErrorBoundary>
        <AppShell />
      </AppErrorBoundary>
    </QueryClientProvider>
  );
}
