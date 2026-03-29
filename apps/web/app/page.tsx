import { CopilotSidebar } from "@copilotkit/react-core/v2";

export default function HomePage() {
  return (
    <main className="flex min-h-screen items-center justify-center px-6 py-16">
      <section className="w-full max-w-3xl rounded-[2rem] border border-white/10 bg-slate-950/45 p-10 shadow-2xl shadow-sky-950/40 backdrop-blur">
        <p className="text-sm font-semibold uppercase tracking-[0.35em] text-sky-300/75">
          GlassBeaker
        </p>
        <h1 className="mt-4 text-4xl font-semibold tracking-tight text-white sm:text-6xl">
          Hello World!
        </h1>
        <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300">
          Tailwind CSS is now available in the Next.js app, so you can build
          UI with utility classes alongside the existing global styles.
        </p>
      </section>
      <CopilotSidebar
        labels={{
          modalHeaderTitle: "GlassBeaker Assistant",
          welcomeMessageText:
            "Ask about the Electron runtime, the standalone Next.js server, or how this starter is wired together.",
          chatInputPlaceholder: "Ask GlassBeaker about this app..."
        }}
      />
    </main>
  );
}
