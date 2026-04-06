export const DEFAULT_PREVIEW_ESM_BASE_URL = "https://esm.sh";

export function resolvePreviewEsmBaseUrl() {
  const value = process.env.NEXT_PUBLIC_PREVIEW_ESM_BASE_URL?.trim() || DEFAULT_PREVIEW_ESM_BASE_URL;

  try {
    return new URL(value).toString().replace(/\/$/, "");
  } catch {
    throw new Error("Invalid NEXT_PUBLIC_PREVIEW_ESM_BASE_URL. It must be an absolute URL.");
  }
}
