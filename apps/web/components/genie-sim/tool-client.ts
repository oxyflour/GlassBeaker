type ToolErrorResult = { error: string };

type JsonResponse = {
  json: () => Promise<unknown>;
  ok: boolean;
  status: number;
};

type FetchLike = (input: string, init: RequestInit) => Promise<JsonResponse>;

function errorMessage(prefix: string, cause: unknown) {
  const detail = cause instanceof Error && cause.message ? cause.message : "unexpected error";
  return `${prefix}: ${detail}`;
}

function detailFromBody(body: unknown) {
  if (!body || typeof body !== "object") {
    return null;
  }
  const detail = (body as { detail?: unknown }).detail;
  return typeof detail === "string" && detail.trim() ? detail : null;
}

export async function postToolJson<T>(
  fetchImpl: FetchLike,
  url: string,
  body: Record<string, unknown>,
  failurePrefix: string,
): Promise<T | ToolErrorResult> {
  try {
    const response = await fetchImpl(url, {
      body: JSON.stringify(body),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    });
    const data = await response.json();
    if (!response.ok) {
      return { error: detailFromBody(data) || `${failurePrefix}: ${response.status}` };
    }
    return data as T;
  } catch (error) {
    return { error: errorMessage(failurePrefix, error) };
  }
}
