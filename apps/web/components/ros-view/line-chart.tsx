import type { RosPlotTopic } from "./demo-data";

const CHART_HEIGHT = 240;
const CHART_WIDTH = 640;
const CHART_PADDING = 24;

export function buildPolylinePoints(values: number[], min: number, max: number): string {
  return values.map((value, index) => {
    const x = CHART_PADDING + (index / Math.max(values.length - 1, 1)) * (CHART_WIDTH - CHART_PADDING * 2);
    const ratio = max === min ? 0.5 : (value - min) / (max - min);
    const y = CHART_HEIGHT - CHART_PADDING - ratio * (CHART_HEIGHT - CHART_PADDING * 2);
    return `${x},${y}`;
  }).join(" ");
}

export default function RosLineChart({ topic }: { topic: RosPlotTopic }) {
  const values = topic.series.flatMap((series) => series.values);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const ticks = [max, (max + min) / 2, min];

  return (
    <div className="flex h-full flex-col bg-[#07131e]">
      <div className="flex h-8 flex-wrap items-center gap-3 border-b border-white/10 px-3 text-[11px] text-slate-400">
        {topic.series.map((series) => (
          <span key={series.label} className="inline-flex items-center gap-2">
            <span className="h-2 w-2" style={{ backgroundColor: series.color }} />
            {series.label}
          </span>
        ))}
        <span className="ml-auto text-cyan-200">{topic.unit}</span>
      </div>
      <div className="relative min-h-0 flex-1 overflow-hidden bg-[linear-gradient(180deg,#0d2234,#060c14)]">
        <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} preserveAspectRatio="none" className="h-full w-full">
          {[0, 1, 2, 3].map((step) => {
            const y = CHART_PADDING + (step / 3) * (CHART_HEIGHT - CHART_PADDING * 2);
            return <line key={step} x1={CHART_PADDING} x2={CHART_WIDTH - CHART_PADDING} y1={y} y2={y} stroke="#173247" strokeWidth="1" />;
          })}
          {topic.series.map((series) => (
            <polyline
              key={series.label}
              fill="none"
              points={buildPolylinePoints(series.values, min, max)}
              stroke={series.color}
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="3"
            />
          ))}
        </svg>
        <div className="pointer-events-none absolute inset-y-4 left-3 flex flex-col justify-between text-[11px] text-slate-500">
          {ticks.map((tick) => <span key={tick}>{tick.toFixed(2)}</span>)}
        </div>
        <div className="pointer-events-none absolute inset-x-4 bottom-3 flex justify-between text-[11px] text-slate-500">
          {topic.timestamps.map((stamp) => <span key={stamp}>{stamp}</span>)}
        </div>
      </div>
    </div>
  );
}
