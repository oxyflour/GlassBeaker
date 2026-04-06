"use client";

import { useCallback, useMemo, useState } from "react";

import { compilePreview } from "./compiler";
import { resolvePreviewEsmBaseUrl, resolvePreviewOrigin } from "./config";
import { HOST_MESSAGE_SOURCE } from "./messages";

export type PreviewFiles = Record<string, string>;

const APP_CODE = `
import { useEffect, useState } from "react";
import Entry from "./entry";
import initialProps from "./props.json";

export default function App() {
  const [props, setProps] = useState(initialProps);

  useEffect(() => {
    function onMessage(event) {
      if (event.data?.source !== ${JSON.stringify(HOST_MESSAGE_SOURCE)} || event.data?.type !== "set-props") {
        return;
      }
      setProps(event.data.props ?? {});
    }

    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);

  return <Entry {...props} />;
}
`;

export function useAgentPreviewState() {
  const [files, setFiles] = useState<PreviewFiles>({});
  const [props, setProps] = useState<unknown>({});

  const hasApp = useMemo(() => Boolean(files["/App.js"]), [files]);

  const setAppCode = useCallback(async (payload: { entry: unknown; files: unknown; props: unknown }) => {
    const nextProps = payload.props ?? {};
    const nextFiles = createPreviewFiles(payload, nextProps);
    await validatePreviewFiles(nextFiles);

    setProps(nextProps);
    setFiles(nextFiles);

    return { ok: true };
  }, []);

  const setAppProps = useCallback((nextProps: unknown) => {
    setProps(nextProps ?? {});
    return { ok: true };
  }, []);

  return { files, hasApp, props, setAppCode, setAppProps };
}

function createPreviewFiles(payload: { entry: unknown; files: unknown; props: unknown }, props: unknown): PreviewFiles {
  const nextFiles = normalizePreviewFiles(payload.files);
  return {
    ...nextFiles,
    "/App.js": APP_CODE,
    "/entry.tsx": typeof payload.entry === "string" ? payload.entry : String(payload.entry ?? ""),
    "/props.json": JSON.stringify(props, null, 2),
  };
}

function normalizePreviewFiles(value: unknown): PreviewFiles {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(value).map(([filePath, content]) => [filePath.replace(/\\/g, "/"), normalizePreviewContent(content)]),
  );
}

function normalizePreviewContent(value: unknown) {
  if (typeof value === "string") {
    return value;
  }
  if (value == null) {
    return "";
  }
  if (typeof value === "object") {
    return JSON.stringify(value, null, 2);
  }
  return String(value);
}

async function validatePreviewFiles(files: PreviewFiles) {
  const compiled = await compilePreview(files, resolvePreviewEsmBaseUrl(), resolvePreviewOrigin());
  compiled.revoke();
}
