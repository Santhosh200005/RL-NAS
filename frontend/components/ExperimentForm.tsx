'use client';

import { useState } from 'react';
import { Method } from '@/lib/api';

export default function ExperimentForm({ method, onRun, disabled }: { method: Method; onRun: (budget: number, seed: number, config: string) => void; disabled?: boolean }) {
  const [budget, setBudget] = useState(20);
  const [seed, setSeed] = useState(0);
  const [config, setConfig] = useState('configs/smoke.yaml');
  return <div className="form-grid">
    <label>Budget<input type="number" min={1} max={200} value={budget} onChange={e => setBudget(Number(e.target.value))} /></label>
    <label>Seed<input type="number" min={0} value={seed} onChange={e => setSeed(Number(e.target.value))} /></label>
    <label className="config-field">Config<select value={config} onChange={e => setConfig(e.target.value)}><option value="configs/smoke.yaml">configs/smoke.yaml</option><option value="configs/default.yaml">configs/default.yaml</option></select></label>
    <button className="run-btn" disabled={disabled} onClick={() => onRun(budget, seed, config)}>{disabled ? 'RUNNING…' : `Start ${method === 'rl' ? 'RL Search' : 'Search'}`}</button>
  </div>;
}
