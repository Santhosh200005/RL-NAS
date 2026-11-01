'use client';

import { useEffect, useMemo, useState } from 'react';
import { api, ArchitectureResult, ExperimentResults, ExperimentStatus, Method } from '@/lib/api';
import ExperimentForm from '@/components/ExperimentForm';
import MetricCard from '@/components/MetricCard';
import LineChart from '@/components/LineChart';
import ParetoChart from '@/components/ParetoChart';
import ArchitectureCard from '@/components/ArchitectureCard';

const METHODS: Method[] = ['rl', 'random', 'bayes', 'rl_acc'];

function methodName(method: string) {
  return ({ rl: 'RL / REINFORCE', random: 'Random Search', bayes: 'Bayesian / EI', rl_acc: 'RL Accuracy-only' } as Record<string, string>)[method] || method;
}

export default function Dashboard() {
  const [backend, setBackend] = useState<'checking' | 'online' | 'offline'>('checking');
  const [experiment, setExperiment] = useState<ExperimentStatus | null>(null);
  const [results, setResults] = useState<ExperimentResults | null>(null);
  const [selectedArch, setSelectedArch] = useState<ArchitectureResult | null>(null);
  const [method, setMethod] = useState<Method>('rl');
  const [error, setError] = useState('');

  useEffect(() => {
    api.health().then(() => setBackend('online')).catch(() => setBackend('offline'));
  }, []);

  useEffect(() => {
    if (!experiment || !['created', 'running'].includes(experiment.status)) return;
    const timer = setInterval(async () => {
      try {
        const status = await api.status(experiment.id);
        setExperiment(status);
        if (status.status === 'done' || status.status === 'failed') {
          clearInterval(timer);
          if (status.status === 'done') setResults(await api.results(experiment.id));
          if (status.error) setError(status.error);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Unable to read experiment status.');
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [experiment?.id, experiment?.status]);

  const run = async (budget: number, seed: number, config: string) => {
    setError('');
    setResults(null);
    setSelectedArch(null);
    try {
      const created = await api.createExperiment({ method, budget, seed, config });
      const started = await api.startExperiment(created.id);
      setExperiment({ id: started.id, status: 'running', episodes_done: 0, budget, error: null });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not start experiment.');
    }
  };

  const best = results?.best;
  const latest = results?.history?.[results.history.length - 1];
  const progress = experiment ? Math.min(100, Math.round((experiment.episodes_done / Math.max(experiment.budget, 1)) * 100)) : 0;
  const rewardSeries = useMemo(() => (results?.history || []).map((r, i) => ({ x: i + 1, y: Number(r.reward) })), [results]);
  const accuracySeries = useMemo(() => (results?.history || []).map((r, i) => ({ x: i + 1, y: Number(r.val_acc) * 100 })), [results]);

  const inspect = async (key: string) => {
    try {
      setSelectedArch(await api.architecture(key));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load architecture.');
    }
  };

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">RESEARCH CONTROL PLANE</div>
          <h1>RL-NAS</h1>
          <p>Reinforcement Learning–Driven Neural Architecture Search</p>
        </div>
        <div className={`connection ${backend}`}><span /> API {backend}</div>
      </header>

      <section className="hero-grid">
        <div className="hero-card">
          <div className="hero-copy">
            <span className="tag">EXPERIMENT RUNNER</span>
            <h2>Search, compare and inspect CNN architectures.</h2>
            <p>Launch the existing FastAPI search service, monitor episodes, and visualize the same reward used by the research pipeline.</p>
          </div>
          <div className="method-tabs">
            {METHODS.map((m) => <button key={m} className={method === m ? 'active' : ''} onClick={() => setMethod(m)}>{methodName(m)}</button>)}
          </div>
          <ExperimentForm method={method} onRun={run} disabled={backend !== 'online' || experiment?.status === 'running'} />
        </div>

        <div className="status-card">
          <div className="card-heading"><span>LIVE RUN</span><strong>{experiment ? experiment.id : '—'}</strong></div>
          {experiment ? <>
            <div className="status-line"><b>{experiment.status.toUpperCase()}</b><span>{experiment.episodes_done} / {experiment.budget} episodes</span></div>
            <div className="progress"><div style={{ width: `${progress}%` }} /></div>
            <div className="mini-grid">
              <MetricCard label="Method" value={methodName(method)} compact />
              <MetricCard label="Seed" value={latest ? String(latest.seed ?? '—') : '—'} compact />
            </div>
          </> : <div className="empty">Configure an experiment and press <b>Start Search</b>.</div>}
          {error && <div className="error">{error}</div>}
        </div>
      </section>

      <section className="metrics">
        <MetricCard label="Best validation accuracy" value={best ? `${(best.val_acc * 100).toFixed(2)}%` : '—'} />
        <MetricCard label="Best reward" value={best ? best.reward.toFixed(4) : '—'} />
        <MetricCard label="Parameters" value={best ? formatNumber(best.params) : '—'} />
        <MetricCard label="FLOPs" value={best ? formatNumber(best.flops) : '—'} />
      </section>

      <section className="charts-grid">
        <div className="panel"><div className="panel-title"><span>REWARD CURVE</span><small>higher is better</small></div><LineChart data={rewardSeries} /></div>
        <div className="panel"><div className="panel-title"><span>VALIDATION ACCURACY</span><small>%</small></div><LineChart data={accuracySeries} /></div>
        <div className="panel wide"><div className="panel-title"><span>PARETO FRONTIER</span><small>parameters vs validation accuracy</small></div><ParetoChart points={(results?.pareto || []).map(r => ({ x: r.params, y: r.val_acc * 100, key: r.key }))} /></div>
      </section>

      <section className="bottom-grid">
        <div className="panel table-panel">
          <div className="panel-title"><span>SEARCH HISTORY</span><small>{results?.history.length || 0} evaluations</small></div>
          {!results?.history.length ? <div className="empty">Results will appear here after the first completed evaluation.</div> : <div className="table-wrap"><table><thead><tr><th>#</th><th>Architecture</th><th>Val acc</th><th>Reward</th><th>Params</th><th>FLOPs</th><th /></tr></thead><tbody>{results.history.map((r, i) => <tr key={`${r.key}-${i}`}><td>{i + 1}</td><td className="mono">{r.key}</td><td>{(r.val_acc * 100).toFixed(2)}%</td><td>{r.reward.toFixed(4)}</td><td>{formatNumber(r.params)}</td><td>{formatNumber(r.flops)}</td><td><button className="link-btn" onClick={() => inspect(r.key)}>Inspect</button></td></tr>)}</tbody></table></div>}
        </div>
        <ArchitectureCard data={selectedArch} />
      </section>

      <footer><span>RL-NAS dashboard</span><span>Frontend only — experiments remain controlled by FastAPI.</span></footer>
    </main>
  );
}

function formatNumber(value: number) {
  if (!Number.isFinite(value)) return '—';
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(Math.round(value));
}
