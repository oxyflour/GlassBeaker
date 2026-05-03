export function OpenInZapdosLink({ sceneUsdaPath }: { sceneUsdaPath: string }) {
  const href = `/demo/zapdos?scene_usd=${encodeURIComponent(sceneUsdaPath)}`;
  return (
    <a
      className="absolute top-3 right-3 z-10 rounded-full border border-slate-700 bg-slate-950/90 px-4 py-2 text-sm text-slate-100 shadow-lg shadow-slate-950/40"
      href={ href }
    >
      Open in Zapdos
    </a>
  );
}
