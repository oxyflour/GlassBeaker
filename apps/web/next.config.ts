import path from "node:path";
import { fileURLToPath } from "node:url";
import type { NextConfig } from "next";

const configDir = path.dirname(fileURLToPath(import.meta.url)),
    bundlerURL = process.env.SANDPACK_BUNDLER || 'https://2-19-8-sandpack.codesandbox.io/'
export default {
    output: "standalone",
    outputFileTracingRoot: path.join(configDir, "../../"),
    allowedDevOrigins: ['dev.yff.me'],
    async rewrites() {
        return {
            // need this to override homepage
            beforeFiles: [{
                source: "/",
                destination: bundlerURL,
                has: [{
                    type: 'host',
                    value: 'dev.yff.me'
                }]
            }],
            // redirect python api
            fallback: [{
                source: "/cors/moonshot/:path*",
                destination: `https://api.moonshot.cn/:path*`
            }, {
                source: "/python/:path*",
                destination: `${process.env.API_REWRITE}/python/:path*`
            }, {
                source: "/:path(.*)",
                destination: `${bundlerURL}:path*`,
                has: [{
                    type: 'host',
                    value: 'dev.yff.me'
                }]
            }]
        }
    }
} satisfies NextConfig;
