import RuntimeCard from "./runtime-card";

export default function HomePage() {
  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">GlassBeaker</p>
        <h1>Next.js UI served inside Electron.</h1>
        <p className="copy">
          The renderer is a regular Next.js app. Electron boots a local server in a
          utility process and loads it just like a browser tab would.
        </p>
      </section>
      <RuntimeCard />
      <section className="notes">
        <h2>What this starter does</h2>
        <ul>
          <li>Uses a `pnpm workspace` monorepo layout.</li>
          <li>Builds Next.js with `output: &quot;standalone&quot;`.</li>
          <li>Starts the bundled Next server from Electron `utilityProcess`.</li>
          <li>Packages the desktop app with `electron-builder`.</li>
        </ul>
      </section>
    </main>
  );
}
