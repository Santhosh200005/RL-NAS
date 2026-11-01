export type Method = 'random' | 'bayes' | 'rl' | 'rl_acc';

export interface ExperimentRequest {
  method: Method;
  budget: number;
  seed: number;
  config: string;
}

export interface ExperimentStatus {
  id: string;
  status: 'created' | 'running' | 'done' | 'failed';
  episodes_done: number;
  budget: number;
  error: string | null;
}

export interface HistoryRow {
  key: string;
  reward: number;
  val_acc: number;
  params: number;
  flops: number;
  latency_ms?: number;
  [key: string]: unknown;
}

export interface ExperimentResults {
  best: HistoryRow | null;
  pareto: HistoryRow[];
  history: HistoryRow[];
}

export interface ArchitectureResult extends HistoryRow {
  arch?: Record<string, unknown>;
}

const API_URL = '/api/backend';

function headers(): HeadersInit {
  return { 'Content-Type': 'application/json' };
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { ...headers(), ...(init.headers || {}) },
    cache: 'no-store'
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ ok: boolean }>('/health'),
  createExperiment: (body: ExperimentRequest) => request<{ id: string; status: string }>('/experiments', { method: 'POST', body: JSON.stringify(body) }),
  startExperiment: (id: string) => request<{ id: string; status: string }>(`/experiments/${id}/start`, { method: 'POST' }),
  status: (id: string) => request<ExperimentStatus>(`/experiments/${id}`),
  results: (id: string) => request<ExperimentResults>(`/experiments/${id}/results`),
  architecture: (key: string) => request<ArchitectureResult>(`/architectures/${encodeURIComponent(key)}`),
  evaluate: (arch: Record<string, unknown>, config = 'configs/smoke.yaml') => request<ArchitectureResult>('/evaluate', { method: 'POST', body: JSON.stringify({ arch, config }) })
};
