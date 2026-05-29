"use client";

import { useCallback, useState } from "react";

import { normalizeIframeUrl, SET_IFRAME_URL_SCHEMA, type SetIframeUrlArgs } from "./iframe-url-tool";

type IframeUrlPreviewProps = {
  className?: string;
  url: string;
};

export function useIframeUrlPreviewState() {
  const [url, setUrl] = useState<string>();

  const setIframeUrl = useCallback((payload: SetIframeUrlArgs) => {
    const parsed = SET_IFRAME_URL_SCHEMA.parse(payload);
    const nextUrl = normalizeIframeUrl(parsed.url, window.location.href);
    setUrl(nextUrl);
    return { ok: true, url: nextUrl };
  }, []);

  return { setIframeUrl, url };
}

export function IframeUrlPreview(props: IframeUrlPreviewProps) {
  return (
    <div className={ ["relative h-full min-h-0 overflow-hidden bg-white", props.className].filter(Boolean).join(" ") }>
      <iframe
        className="h-full w-full border-0 bg-white"
        referrerPolicy="no-referrer-when-downgrade"
        sandbox="allow-downloads allow-forms allow-modals allow-popups allow-same-origin allow-scripts"
        src={ props.url }
        title="Agent iframe preview"
      />
    </div>
  );
}
