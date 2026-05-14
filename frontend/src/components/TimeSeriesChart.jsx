import { useDeferredValue } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import useStore from '../store/useStore';

const createPlotly =
  typeof createPlotlyComponent === 'function'
    ? createPlotlyComponent
    : createPlotlyComponent?.default;

const Plot = typeof createPlotly === 'function' ? createPlotly(Plotly) : null;

export default function TimeSeriesChart() {
  const { liveMetrics, anomalies } = useStore();
  const deferredMetrics = useDeferredValue(liveMetrics);
  const deferredAnomalies = useDeferredValue(anomalies);
  const safeMetrics = Array.isArray(deferredMetrics) ? deferredMetrics.filter(Boolean) : [];
  const safeAnomalies = Array.isArray(deferredAnomalies) ? deferredAnomalies.filter(Boolean) : [];

  const timestamps = safeMetrics.map((metric) => new Date(metric.timestamp));
  const temperature = safeMetrics.map((metric) => Number(metric.temperature_k));
  const rollingMean = safeMetrics.map((metric) => Number(metric.rolling_mean_16 ?? 0));
  const vibration = safeMetrics.map((metric) => Number(metric.vibration_mms));

  const tempByTimestamp = {};
  for (const metric of safeMetrics) {
    tempByTimestamp[metric.timestamp] = metric.temperature_k;
  }

  const anomalyTimes = safeAnomalies.map((item) => new Date(item.timestamp));
  const anomalyValues = safeAnomalies.map((item) => tempByTimestamp[item.timestamp] ?? null);

  if (!Plot) {
    return (
      <div className="h-[420px] w-full rounded-2xl border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">
        Plot component failed to initialize. Restart frontend after dependency refresh.
      </div>
    );
  }

  return (
    <Plot
      data={[
        {
          x: timestamps,
          y: temperature,
          type: 'scatter',
          mode: 'lines',
          name: 'Temperature',
          line: { color: '#67e8f9', width: 2.5 },
        },
        {
          x: timestamps,
          y: rollingMean,
          type: 'scatter',
          mode: 'lines',
          name: 'Rolling Mean',
          line: { color: '#f59e0b', width: 1.4, dash: 'dot' },
        },
        {
          x: timestamps,
          y: vibration,
          type: 'scatter',
          mode: 'lines',
          name: 'Vibration',
          yaxis: 'y2',
          line: { color: '#34d399', width: 1.8 },
        },
        {
          x: anomalyTimes,
          y: anomalyValues,
          type: 'scatter',
          mode: 'markers',
          name: 'Anomalies',
          marker: { color: '#fb7185', size: 10, symbol: 'diamond' },
        },
      ]}
      layout={{
        autosize: true,
        margin: { t: 18, r: 48, b: 42, l: 50 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#cbd5e1', family: 'IBM Plex Sans, sans-serif' },
        xaxis: {
          gridcolor: 'rgba(71,85,105,0.35)',
          zerolinecolor: 'rgba(71,85,105,0.2)',
          title: 'Time',
        },
        yaxis: {
          gridcolor: 'rgba(71,85,105,0.35)',
          zerolinecolor: 'rgba(71,85,105,0.2)',
          title: 'Temperature (K)',
        },
        yaxis2: {
          title: 'Vibration (mm/s)',
          overlaying: 'y',
          side: 'right',
          showgrid: false,
        },
        legend: {
          orientation: 'h',
          x: 0,
          y: 1.12,
          bgcolor: 'rgba(2,6,23,0.0)',
        },
      }}
      config={{ responsive: true, displayModeBar: false }}
      className="h-[420px] w-full"
      useResizeHandler
    />
  );
}
