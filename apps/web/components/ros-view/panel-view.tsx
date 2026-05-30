import type { RosTopic } from "./demo-data";
import RosLineChart from "./line-chart";

function toneClasses(tone?: "good" | "warn" | "alert"): string {
  if (tone === "good") return "text-emerald-200";
  if (tone === "warn") return "text-amber-100";
  return "text-rose-100";
}

function TopicBody({ topic }: { topic: RosTopic }) {
  if (topic.kind === "image") {
    return <img src={topic.src} alt={topic.label} className="h-full w-full object-cover" />;
  }
  if (topic.kind === "plot") {
    return <RosLineChart topic={topic} />;
  }
  return (
    <div className="grid h-full grid-cols-2 text-xs text-slate-300">
      {topic.fields.map((field) => (
        <div key={field.label} className="border-r border-b border-white/10 px-3 py-2 even:border-r-0">
          <div className="text-[10px] uppercase tracking-[0.18em] text-slate-500">{field.label}</div>
          <div className={`mt-2 text-lg font-semibold ${toneClasses(field.tone)}`}>{field.value}</div>
        </div>
      ))}
    </div>
  );
}

export default function RosPanelView({
  topic,
  topics,
  canClose,
  onTopicChange,
  onSplitRight,
  onSplitDown,
  onClose,
}: {
  topic: RosTopic;
  topics: RosTopic[];
  canClose: boolean;
  onTopicChange: (topicId: string) => void;
  onSplitRight: () => void;
  onSplitDown: () => void;
  onClose: () => void;
}) {
  const topicSelectionDisabled = topics.length === 1 && topics[0]?.id === "__empty__";
  return (
    <section className="flex h-full min-h-0 flex-col border border-white/10 bg-[#04090f]">
      <div className="flex h-9 items-center gap-2 border-b border-white/10 bg-[#0b141c] px-2 text-xs text-slate-300">
        <select
          value={topic.id}
          onChange={(event) => onTopicChange(event.target.value)}
          className="min-w-0 flex-1 bg-transparent outline-none"
          disabled={topicSelectionDisabled}
        >
          {topics.map((option) => <option key={option.id} value={option.id} className="bg-slate-950">{option.label}</option>)}
        </select>
        <button type="button" onClick={onSplitRight} className="border border-white/10 px-2 text-[11px] text-slate-200">
          Split Right
        </button>
        <button type="button" onClick={onSplitDown} className="border border-white/10 px-2 text-[11px] text-slate-200">
          Split Down
        </button>
        <button type="button" onClick={onClose} className="border border-white/10 px-2 text-[11px] text-slate-200 disabled:opacity-40" disabled={!canClose}>
          Close
        </button>
      </div>
      <div className="min-h-0 flex-1">
        <TopicBody topic={topic} />
      </div>
    </section>
  );
}
