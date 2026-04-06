import type { PiFrontendToolResultPayload } from "../../../../components/agent/pi/protocol";

import { resolveFrontendToolCall } from "../frontend-tools";

export const runtime = "nodejs";

export async function POST(request: Request) {
  let body: PiFrontendToolResultPayload;

  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  if (!body?.requestId || !body?.toolCallId) {
    return Response.json({ error: "Invalid frontend tool payload." }, { status: 400 });
  }

  if (!resolveFrontendToolCall(body)) {
    return Response.json({ error: "Frontend tool call not found." }, { status: 404 });
  }

  return Response.json({ ok: true });
}
