export default function MetricCard({ label, value, compact }: { label: string; value: string; compact?: boolean }) {
  return <div className={`metric-card ${compact ? 'compact' : ''}`}><span>{label}</span><strong>{value}</strong></div>;
}
