import path from "node:path";
import { fileURLToPath } from "node:url";
import type { NextConfig } from "next";

const configDir = path.dirname(fileURLToPath(import.meta.url));

const nextConfig = {
  output: "standalone",
  outputFileTracingRoot: path.join(configDir, "../../"),
  serverExternalPackages: [
    "@mariozechner/pi-coding-agent",
    "@mariozechner/pi-tui",
    "@mariozechner/clipboard",
    "koffi",
  ],
  async rewrites() {
    return {
      fallback: [
        {
          source: "/cors/moonshot/:path*",
          destination: "https://api.moonshot.cn/:path*",
        },
        {
          source: "/python/:path*",
          destination: `${process.env.API_REWRITE || "http://localhost:13001/"}api/:path*`,
        },
      ],
    };
  },
} satisfies NextConfig;

export default nextConfig;
