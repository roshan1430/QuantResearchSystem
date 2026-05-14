import useStore from '../store/useStore';

class WebSocketService {
  constructor() {
    this.socket = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 8;
    this.metricBuffer = [];
    this.anomalyBuffer = [];
    this.flushTimer = null;
  }

  connect(url) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      return;
    }

    this.socket = new WebSocket(url);

    this.socket.onopen = () => {
      useStore.getState().setConnected(true);
      this.reconnectAttempts = 0;
    };

    this.socket.onmessage = (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch (error) {
        console.error('WebSocket payload parse error:', error);
        return;
      }
      if (!message || typeof message !== 'object') {
        return;
      }
      if (message.type === 'metric' && message.data) {
        this.metricBuffer.push(message.data);
      }
      if (message.type === 'anomaly' && message.data) {
        this.anomalyBuffer.push(message.data);
      }
      this.scheduleFlush();
    };

    this.socket.onclose = () => {
      useStore.getState().setConnected(false);
      this.reconnect(url);
    };
  }

  reconnect(url) {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      return;
    }
    this.reconnectAttempts += 1;
    setTimeout(() => this.connect(url), this.reconnectAttempts * 1500);
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
    }
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
  }

  scheduleFlush() {
    if (this.flushTimer) {
      return;
    }
    this.flushTimer = setTimeout(() => {
      const metrics = this.metricBuffer.splice(0, this.metricBuffer.length);
      const anomalies = this.anomalyBuffer.splice(0, this.anomalyBuffer.length);
      if (metrics.length) {
        useStore.getState().addMetricsBatch(metrics);
      }
      if (anomalies.length) {
        useStore.getState().addAnomaliesBatch(anomalies.reverse());
      }
      this.flushTimer = null;
    }, 250);
  }
}

const wsService = new WebSocketService();
export default wsService;
