import { readFile } from "node:fs/promises";

import type { NextRequest } from "next/server";

import { resolvePreviewPackageRequest } from "../request";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(_request: NextRequest, context: { params: Promise<{ specifier?: string[] }> }) {
  try {
    const { specifier } = await context.params;
    const resolved = resolvePreviewPackageRequest(specifier);
    const body = await readFile(resolved.filePath);
    return new Response(body, {
      headers: {
        "Cache-Control": "no-store",
        "Content-Type": resolved.contentType,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to resolve preview package.";
    return new Response(message, {
      status: 404,
      headers: {
        "Cache-Control": "no-store",
        "Content-Type": "text/plain; charset=utf-8",
      },
    });
  }
}
