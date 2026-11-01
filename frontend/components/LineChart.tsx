export default function LineChart({ data }: { data: { x: number; y: number }[] }) {
  if (!data.length) return <div className="chart-empty">Run an experiment to populate this chart.</div>;
  const w = 720, h = 250, p = 30;
  const ys = data.map(d => d.y), min = Math.min(...ys), max = Math.max(...ys), range = max - min || 1;
  const pts = data.map((d, i) => `${p + (i / Math.max(data.length - 1, 1)) * (w - p * 2)},${h - p - ((d.y - min) / range) * (h - p * 2)}`).join(' ');
  return <svg viewBox={`0 0 ${w} ${h}`} className="chart" role="img" aria-label="line chart"><line x1={p} y1={h-p} x2={w-p} y2={h-p} className="axis"/><polyline points={pts} fill="none" className="series"/><text x={p} y={18} className="chart-label">max {max.toFixed(3)}</text><text x={p} y={h-5} className="chart-label">min {min.toFixed(3)}</text></svg>;
}
