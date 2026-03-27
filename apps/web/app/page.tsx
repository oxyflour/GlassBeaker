import { CopilotSidebar } from "@copilotkit/react-core/v2";

export default function HomePage() {
  return <>
    Hello World!
    <CopilotSidebar
      labels={{
        modalHeaderTitle: "GlassBeaker Assistant",
        welcomeMessageText:
          "Ask about the Electron runtime, the standalone Next.js server, or how this starter is wired together.",
        chatInputPlaceholder: "Ask GlassBeaker about this app..."
      }}
    />
  </>
}
