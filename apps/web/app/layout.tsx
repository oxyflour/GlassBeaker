import type { Metadata } from "next";
import type { ReactNode } from "react";
import { CopilotKit } from "@copilotkit/react-core";

import "@copilotkit/react-ui/v2/styles.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "GlassBeaker",
  description: "Minimal Next.js + Electron workspace"
};

type RootLayoutProps = Readonly<{
  children: ReactNode;
}>;

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en" className="dark">
      <body>
        <CopilotKit runtimeUrl="/api/copilotkit" agent="default">
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}
