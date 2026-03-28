import path from "node:path";
import { fileURLToPath } from "node:url";
import type { NextConfig } from "next";

const configDir = path.dirname(fileURLToPath(import.meta.url)),
  apiRewrite = process.env.API_REWRITE || 'http://127.0.0.1:4000'
export default {
  output: "standalone",
  outputFileTracingRoot: path.join(configDir, "../../"),
  async rewrites() {
    return {
      fallback: [
        {
          source: "/api/:path*",
          destination: `${apiRewrite}/api/:path*`
        }
      ]
    };
  }
} satisfies NextConfig;
