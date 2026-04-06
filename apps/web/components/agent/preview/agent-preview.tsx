"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { compilePreview } from "./compiler";
import { resolvePreviewEsmBaseUrl } from "./config";
import { createPreviewHostMessage, isPreviewFrameMessage } from "./messages";
import type { PreviewFiles } from "./state";

const EMPTY_SRC_DOC = "<!doctype html><html><body></body></html>";

type PreviewStatus = "building" | "error" | "idle" | "loading" | "ready";
type EsmConfig = { error: string; kind: "error" } | { kind: "ready"; value: string };

type AgentPreviewProps = {
  className?: string;
  files: PreviewFiles;
  props: unknown;
};

export function AgentPreview(props: AgentPreviewProps) {
  const { className, files, props: previewProps } = props;
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const revokeRef = useRef<(() => void) | null>(null);
  const [buildId, setBuildId] = useState(0);
  const [error, setError] = useState<string>();
  const [srcDoc, setSrcDoc] = useState(EMPTY_SRC_DOC);
  const [status, setStatus] = useState<PreviewStatus>("idle");

  const esmConfig = useMemo(resolveEsmConfig, []);
  const hasApp = Boolean(files["/App.js"] || files["App.js"]);

  const postProps = useCallback(() => {
    iframeRef.current?.contentWindow?.postMessage(createPreviewHostMessage(previewProps), "*");
  }, [previewProps]);

  useEffect(() => () => revokeRef.current?.(), []);

  useEffect(() => {
    if (!hasApp) {
      setError(undefined);
      setStatus("idle");
      return;
    }
    if (esmConfig.kind === "error") {
      setError(esmConfig.error);
      setStatus("error");
      return;
    }

    let cancelled = false;
    setError(undefined);
    setStatus("building");

    void compilePreview(files, esmConfig.value)
      .then((compiled) => {
        if (cancelled) {
          compiled.revoke();
          return;
        }

        revokeRef.current?.();
        revokeRef.current = compiled.revoke;
        setSrcDoc(compiled.srcDoc);
        setBuildId((current) => current + 1);
        setStatus("loading");
      })
      .catch((nextError) => {
        if (cancelled) {
          return;
        }
        setError(nextError instanceof Error ? nextError.message : String(nextError));
        setStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [esmConfig, files, hasApp]);

  useEffect(() => {
    function onMessage(event: MessageEvent) {
      if (event.source !== iframeRef.current?.contentWindow || !isPreviewFrameMessage(event.data)) {
        return;
      }
      if (event.data.type === "ready") {
        setError(undefined);
        setStatus("ready");
        return;
      }
      setError(event.data.error || "Unknown preview error.");
      setStatus("error");
    }

    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);

  useEffect(() => {
    if (status === "ready") {
      postProps();
    }
  }, [buildId, postProps, status]);

  return (
    <div className={ ["relative h-full min-h-0 overflow-hidden bg-white", className].filter(Boolean).join(" ") }>
      <iframe
        key={ buildId }
        ref={ iframeRef }
        className="h-full w-full border-0 bg-white"
        sandbox="allow-same-origin allow-scripts"
        srcDoc={ srcDoc }
        title="Agent preview"
      />
      { status !== "ready" ? <PreviewOverlay error={ error } status={ status } /> : null }
    </div>
  );
}

function PreviewOverlay(props: { error?: string; status: PreviewStatus }) {
  const { error, status } = props;
  const message = error || (status === "loading" ? "Loading preview..." : "Building preview...");

  return (
    <div className="absolute inset-0 flex items-center justify-center bg-white/92 px-6 text-sm text-slate-700">
      <div className="max-w-xl whitespace-pre-wrap rounded border border-slate-200 bg-white px-4 py-3 shadow-sm">
        { message }
      </div>
    </div>
  );
}

function resolveEsmConfig(): EsmConfig {
  try {
    return { kind: "ready", value: resolvePreviewEsmBaseUrl() };
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error), kind: "error" };
  }
}
