import { ArchitectureResult } from '@/lib/api';

export default function ArchitectureCard({ data }: { data: ArchitectureResult | null }) {
  return <div className="panel architecture-panel"><div className="panel-title"><span>ARCHITECTURE INSPECTOR</span><small>{data?.key || 'select a row'}</small></div>{!data ? <div className="empty">Select <b>Inspect</b> on a search result to view its logged architecture metrics.</div> : <div className="arch-body"><div className="arch-metrics"><div><span>Validation</span><b>{(data.val_acc * 100).toFixed(2)}%</b></div><div><span>Reward</span><b>{data.reward.toFixed(4)}</b></div><div><span>Latency</span><b>{data.latency_ms != null ? `${Number(data.latency_ms).toFixed(2)} ms` : '—'}</b></div></div><pre>{JSON.stringify(data.arch || data, null, 2)}</pre></div>}</div>;
}
