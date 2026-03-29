import type { Metadata } from "next";
import type { ReactNode } from "react";
import { CopilotKit } from "@copilotkit/react-core";

import "@copilotkit/react-ui/v2/styles.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "GlassBeaker",
  description: "Minimal Next.js + Electron workspace"
};

export default function RootLayout({ children }: any) {
  return (
    <html lang="en">
      <body>
        <CopilotKit
          showDevConsole={ false }
          enableInspector={ false }
          runtimeUrl="/api/copilotkit"
          agent="default">
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}
