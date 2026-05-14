import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchAnomalies, fetchForecast, fetchHealth, fetchMetrics, fetchModels } from '../services/api';
import useStore from '../store/useStore';

export default function usePlatformData() {
  const {
    setInitialMetrics,
    setInitialAnomalies,
    setForecast,
    setCatalog,
    setHealth,
    setApiStatus,
    liveMetrics,
  } = useStore();

  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 15000,
  });

  const metricsQuery = useQuery({
    queryKey: ['metrics'],
    queryFn: fetchMetrics,
    refetchInterval: 10000,
  });

  const anomaliesQuery = useQuery({
    queryKey: ['anomalies'],
    queryFn: fetchAnomalies,
    refetchInterval: 12000,
  });

  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: fetchModels,
    staleTime: 30000,
  });

  const forecastQuery = useQuery({
    queryKey: ['forecast'],
    queryFn: () => fetchForecast(liveMetrics.slice(-24)),
    enabled: liveMetrics.length >= 16,
    refetchInterval: 20000,
  });

  useEffect(() => {
    if (metricsQuery.data?.points) {
      setInitialMetrics(metricsQuery.data.points);
    }
  }, [metricsQuery.data, setInitialMetrics]);

  useEffect(() => {
    if (anomaliesQuery.data) {
      setInitialAnomalies(anomaliesQuery.data);
    }
  }, [anomaliesQuery.data, setInitialAnomalies]);

  useEffect(() => {
    if (modelsQuery.data?.models) {
      setCatalog(modelsQuery.data.models);
    }
  }, [modelsQuery.data, setCatalog]);

  useEffect(() => {
    if (forecastQuery.data?.forecast) {
      setForecast(forecastQuery.data.forecast);
    }
  }, [forecastQuery.data, setForecast]);

  useEffect(() => {
    if (healthQuery.data) {
      setHealth(healthQuery.data);
    }
  }, [healthQuery.data, setHealth]);

  useEffect(() => {
    setApiStatus('health', healthQuery.isError ? 'error' : healthQuery.isFetching ? 'loading' : 'ok');
    setApiStatus('metrics', metricsQuery.isError ? 'error' : metricsQuery.isFetching ? 'loading' : 'ok');
    setApiStatus('anomalies', anomaliesQuery.isError ? 'error' : anomaliesQuery.isFetching ? 'loading' : 'ok');
    setApiStatus('models', modelsQuery.isError ? 'error' : modelsQuery.isFetching ? 'loading' : 'ok');
    setApiStatus('forecast', forecastQuery.isError ? 'error' : forecastQuery.isFetching ? 'loading' : 'ok');
  }, [
    anomaliesQuery.isError,
    anomaliesQuery.isFetching,
    forecastQuery.isError,
    forecastQuery.isFetching,
    healthQuery.isError,
    healthQuery.isFetching,
    metricsQuery.isError,
    metricsQuery.isFetching,
    modelsQuery.isError,
    modelsQuery.isFetching,
    setApiStatus,
  ]);
}
