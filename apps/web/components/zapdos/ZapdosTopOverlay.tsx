'use client'

import { useState } from "react";

import { CameraOverrideSaveButton } from "./CameraOverrideSaveButton";
import { SpaceMouseModeSelect } from "./SpaceMouseModeSelect";

export function ZapdosTopOverlay({
  defaultSettingsOpen = false,
  mode = "translate",
  selectedBody = null,
  sess,
  sse,
}: {
  defaultSettingsOpen?: boolean
  mode?: "translate" | "rotate"
  selectedBody?: string | null
  sess: string
  sse: number
}) {
  const [open, setOpen] = useState(defaultSettingsOpen);

  return <>
    <div className="absolute left-8 top-8">
      <div className="rounded-md bg-black/60 px-3 py-2 text-white backdrop-blur-sm">
        SSE { sse.toFixed(2) } Hz
        <div>Selected { selectedBody ?? "none" }</div>
        <div>Mode { mode }</div>
      </div>
    </div>
    <div className="absolute right-8 top-8">
      <div className="relative">
        <button
          aria-expanded={ open }
          aria-haspopup="dialog"
          aria-label={ open ? "Close settings" : "Open settings" }
          className="rounded-md border border-white/20 bg-black/60 px-3 py-2 text-sm text-white backdrop-blur-sm"
          onClick={ () => setOpen(current => !current) }
          type="button">
          Config
        </button>
        { open ? <div className="absolute right-0 mt-3 flex w-72 max-w-[calc(100vw-4rem)] flex-col gap-3">
          <SpaceMouseModeSelect />
          <CameraOverrideSaveButton sess={ sess } />
        </div> : null }
      </div>
    </div>
  </>;
}
