import type { NextRequest } from "next/server";

import { resolvePreviewModuleRequest } from "../request";
import { buildPreviewModuleSource } from "../source";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(request: NextRequest, context: { params: Promise<{ specifier?: string[] }> }) {
  try {
    const { specifier } = await context.params;
    const resolved = resolvePreviewModuleRequest(specifier);
    const code = await buildPreviewModuleSource(resolved, request.nextUrl.origin);
    return createJavaScriptResponse(code);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to build preview module.";
    return createJavaScriptResponse(`throw new Error(${JSON.stringify(message)});\n`);
  }
}

function createJavaScriptResponse(body: string) {
  return new Response(body, {
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": "application/javascript; charset=utf-8",
    },
  });
}
