export function SceneUsdaBadge({ path }: { path: string }) {
  return (
    <div
      className="pointer-events-none absolute bottom-3 left-3 z-10 max-w-[72%] rounded-lg border border-slate-700/70 bg-slate-950/88 px-3 py-2 text-xs text-slate-300 shadow-lg shadow-slate-950/40"
      data-slot="usda-badge"
      title={ path }
    >
      <div className="mb-1 text-[10px] font-semibold tracking-[0.18em] text-slate-500 uppercase">USDA</div>
      <div className="truncate font-mono text-slate-100">{path}</div>
    </div>
  );
}
