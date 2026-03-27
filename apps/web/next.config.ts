import path from "node:path";
import { fileURLToPath } from "node:url";
import type { NextConfig } from "next";

const configDir = path.dirname(fileURLToPath(import.meta.url));

function getPythonOrigin(): string {
  const explicitOrigin = process.env.GLASSBEAKER_PYTHON_ORIGIN?.trim();

  if (explicitOrigin) {
    return explicitOrigin;
  }

  const host = process.env.GLASSBEAKER_PYTHON_HOST?.trim() || "127.0.0.1";
  const port = process.env.GLASSBEAKER_PYTHON_PORT?.trim() || "8000";

  return `http://${host}:${port}`;
}

const nextConfig: NextConfig = {
  output: "standalone",
  outputFileTracingRoot: path.join(configDir, "../../"),
  async rewrites() {
    return {
      fallback: [
        {
          source: "/api/:path*",
          destination: `${getPythonOrigin()}/api/:path*`
        }
      ]
    };
  }
};

export default nextConfig;
