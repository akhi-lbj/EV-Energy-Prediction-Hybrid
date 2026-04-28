'use client';

import React, { useState } from 'react';

export default function Home() {
  const [sessionData, setSessionData] = useState<any>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const fetchRandomSession = async () => {
    setIsLoading(true);
    setError('');
    setPrediction(null); // Reset prediction when fetching new session
    try {
      const res = await fetch('/api/predict?mode=fetch');
      if (!res.ok) throw new Error('Failed to fetch session');
      const data = await res.json();
      setSessionData(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const runLivePrediction = async () => {
    if (!sessionData) return;
    setIsLoading(true);
    setError('');
    try {
      const res = await fetch(`/api/predict?mode=predict&idx=${sessionData.row_idx}`);
      if (!res.ok) throw new Error('Failed to run prediction');
      const data = await res.json();
      setPrediction(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--spacing-xl)' }}>
        <h1 className="display-title" style={{ marginBottom: 0 }}>Smart EV Predictor</h1>
        <button onClick={fetchRandomSession} className="btn-randomizer" disabled={isLoading}>
          ⚡ Auto-Fill Random Session
        </button>
      </div>

      <div className="grid-2">
        {/* Input Panel */}
        <div className="card">
          <h2 className="headline">Session Parameters (Read-Only)</h2>
          
          {sessionData ? (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--spacing-md)', marginBottom: 'var(--spacing-lg)' }}>
                <div>
                  <label className="label-text">Requested Energy (kWh)</label>
                  <div className="form-input" style={{ opacity: 0.7 }}>{sessionData.parsed_kWhRequested.toFixed(2)}</div>
                </div>
                <div>
                  <label className="label-text">Minutes Available</label>
                  <div className="form-input" style={{ opacity: 0.7 }}>{sessionData.parsed_minutesAvailable}</div>
                </div>
                <div>
                  <label className="label-text">Miles Requested</label>
                  <div className="form-input" style={{ opacity: 0.7 }}>{sessionData.parsed_milesRequested.toFixed(1)}</div>
                </div>
                <div>
                  <label className="label-text">Wh Per Mile</label>
                  <div className="form-input" style={{ opacity: 0.7 }}>{sessionData.parsed_WhPerMile.toFixed(1)}</div>
                </div>
                <div>
                  <label className="label-text">Connection Time</label>
                  <div className="form-input" style={{ opacity: 0.7, fontSize: '14px' }}>
                    {new Date(sessionData.connectionTime).toLocaleString()}
                  </div>
                </div>
                <div>
                  <label className="label-text">Day of Week</label>
                  <div className="form-input" style={{ opacity: 0.7 }}>{sessionData.day_of_week}</div>
                </div>
                <div>
                  <label className="label-text">Revision Count</label>
                  <div className="form-input" style={{ opacity: 0.7 }}>{sessionData.revisionCount}</div>
                </div>
                <div>
                  <label className="label-text">Urgency Score</label>
                  <div className="form-input" style={{ opacity: 0.7 }}>{sessionData.urgency_score.toFixed(1)}</div>
                </div>
                <div>
                  <label className="label-text">Flexibility Index</label>
                  <div className="form-input" style={{ opacity: 0.7 }}>{sessionData.flexibility_index.toFixed(2)}</div>
                </div>
                <div>
                  <label className="label-text">Habit Stability</label>
                  <div className="form-input" style={{ opacity: 0.7 }}>{sessionData.habit_stability.toFixed(2)}</div>
                </div>
              </div>

              <button 
                onClick={runLivePrediction} 
                className="btn-primary" 
                style={{ width: '100%' }}
                disabled={isLoading}
              >
                {isLoading ? 'Running Prediction...' : 'Run Live Prediction'}
              </button>
            </>
          ) : (
            <div style={{ padding: '40px', textAlign: 'center', opacity: 0.5 }}>
              Click "Auto-Fill Random Session" to load data from the ACN dataset.
            </div>
          )}

          {error && <p style={{ color: 'var(--color-error)', marginTop: 'var(--spacing-md)' }}>{error}</p>}
        </div>

        {/* Output Panel */}
        <div className="card">
          <h2 className="headline">Probabilistic Forecast</h2>
          
          <div className="data-panel">
            <label className="label-text">Expected Delivery (Median)</label>
            <div className="data-readout">
              {prediction ? `${prediction.pred_median.toFixed(2)} kWh` : '---'}
            </div>
          </div>

          <div className="data-panel" style={{ borderLeftColor: 'var(--color-secondary)' }}>
            <label className="label-text">90% Confidence Interval (Safe Bounds)</label>
            <div className="data-readout" style={{ fontSize: '24px', color: 'var(--color-text-main)' }}>
              {prediction ? `[${prediction.lower_conformal.toFixed(2)} - ${prediction.upper_conformal.toFixed(2)}] kWh` : '---'}
            </div>
          </div>

          <div className="data-panel" style={{ borderLeftColor: 'var(--color-error)' }}>
            <label className="label-text">True Value (From Dataset)</label>
            <div className="data-readout" style={{ fontSize: '24px', color: 'var(--color-text-main)' }}>
              {prediction && sessionData ? `${sessionData.true_kWhDelivered.toFixed(2)} kWh` : '---'}
            </div>
          </div>

        </div>
      </div>
    </main>
  );
}
