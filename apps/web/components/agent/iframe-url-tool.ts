import { z } from "zod";

import type { PiFrontendToolParameter } from "./pi/protocol";

export const SET_IFRAME_URL_TOOL_NAME = "set_iframe_url";

export const SET_IFRAME_URL_DESCRIPTION =
  "Set the live iframe to an existing page URL. Use an absolute http(s) URL or a same-origin path.";

export const SET_IFRAME_URL_SCHEMA = z.object({
  url: z
    .string()
    .trim()
    .min(1)
    .describe("Iframe URL to open. Use an absolute http(s) URL or a same-origin path starting with /."),
});

export type SetIframeUrlArgs = z.infer<typeof SET_IFRAME_URL_SCHEMA>;

export const SET_IFRAME_URL_PARAMETERS: PiFrontendToolParameter[] = [
  {
    name: "url",
    type: "string",
    description: "Iframe URL to open. Use an absolute http(s) URL or a same-origin path starting with /.",
    required: true,
  },
];

export const IFRAME_URL_ADDITIONAL_INSTRUCTIONS = [
  "Use `set_iframe_url` to show an existing browser page in the live iframe.",
  "The `url` must be an absolute http(s) URL or a same-origin path.",
  "Do not generate React source payloads or call removed preview-code tools.",
].join("\n");

export function normalizeIframeUrl(rawUrl: string, baseHref: string) {
  const value = rawUrl.trim();
  const url = new URL(value, baseHref);

  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("Iframe URL must use http or https.");
  }

  if (value.startsWith("/") && !value.startsWith("//")) {
    return `${url.pathname}${url.search}${url.hash}`;
  }
  return url.toString();
}
