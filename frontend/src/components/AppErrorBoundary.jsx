import React from 'react';

export default class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error) {
    // Keep this log so browser console shows the root cause when rendering fails.
    console.error('Frontend runtime error:', error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-background text-foreground p-8">
          <div className="mx-auto max-w-3xl rounded-2xl border border-rose-500/40 bg-rose-500/10 p-6">
            <h1 className="text-xl font-semibold text-rose-200">Frontend runtime error</h1>
            <p className="mt-2 text-sm text-rose-100/90">
              The UI recovered into safe mode. Refresh once, and if it repeats, share browser console errors.
            </p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
