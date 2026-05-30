'use client'

import type { ChangeEvent } from "react";
import { useEffect, useState } from "react";

import { createSpaceMouseModeRequest, deriveSpaceMouseMode, type SpaceMouseMode, type SpaceMouseStatus } from "./spacemouse-mode";

const STATUS_URL = "/python/teleop/spacemouse/status";
const MODE_URL = "/python/teleop/spacemouse/set_mode";

export function SpaceMouseModeSelect() {
  const [mode, setMode] = useState<SpaceMouseMode>("off");
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    void readSpaceMouseMode()
      .then((next) => {
        if (cancelled) return;
        setMode(next);
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setError(describeError(cause));
      })
      .finally(() => {
        if (!cancelled) setBusy(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleChange(event: ChangeEvent<HTMLSelectElement>) {
    const next = event.target.value as SpaceMouseMode;
    const previous = mode;
    setMode(next);
    setBusy(true);
    setError("");
    try {
      setMode(await writeSpaceMouseMode(next));
    } catch (cause) {
      setMode(previous);
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  }

  return <div className="rounded-md bg-black/60 px-3 py-2 text-white backdrop-blur-sm">
    <label className="mr-2 text-sm" htmlFor="spacemouse-mode">SpaceMouse</label>
    <select
      id="spacemouse-mode"
      className="rounded border border-white/20 bg-black/40 px-2 py-1 text-sm"
      disabled={ busy }
      onChange={ event => void handleChange(event) }
      value={ mode }>
      <option value="off">关闭</option>
      <option value="left">左臂</option>
      <option value="right">右臂</option>
    </select>
    { error ? <div className="mt-2 max-w-56 text-xs text-red-200">{ error }</div> : null }
  </div>
}

async function readSpaceMouseMode(): Promise<SpaceMouseMode> {
  const response = await fetch(STATUS_URL);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const status = await response.json() as SpaceMouseStatus;
  return deriveSpaceMouseMode(status);
}

async function writeSpaceMouseMode(mode: SpaceMouseMode): Promise<SpaceMouseMode> {
  const response = await fetch(MODE_URL, createSpaceMouseModeRequest(mode));
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const status = await response.json() as SpaceMouseStatus;
  return deriveSpaceMouseMode(status);
}

function describeError(cause: unknown): string {
  return cause instanceof Error ? cause.message : "SpaceMouse unavailable";
}
