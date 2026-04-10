export const DEFAULT_PREVIEW_ESM_BASE_URL = "https://esm.sh";
export const PREVIEW_MODULE_ROUTE_PATH = "/api/preview/module";
export const PREVIEW_PACKAGE_ROUTE_PATH = "/api/preview/package";

export function resolvePreviewEsmBaseUrl() {
  const value = process.env.NEXT_PUBLIC_PREVIEW_ESM_BASE_URL?.trim() || DEFAULT_PREVIEW_ESM_BASE_URL;

  try {
    return new URL(value).toString().replace(/\/$/, "");
  } catch {
    throw new Error("Invalid NEXT_PUBLIC_PREVIEW_ESM_BASE_URL. It must be an absolute URL.");
  }
}

export function resolvePreviewOrigin() {
  if (typeof window === "undefined" || !window.location.origin) {
    throw new Error("Preview origin is unavailable outside the browser.");
  }

  return window.location.origin;
}
