'use client'

import { useState } from "react";

import { AddBenchmarkTableButton } from "./debug/AddBenchmarkTableButton";
import { GrabTheAppleButton } from "./debug/GrabTheAppleButton";
import { PlaceTheAppleButton } from "./debug/PlaceTheAppleButton";
import { CameraOverrideSaveButton } from "./settings/CameraOverrideSaveButton";
import { RobotModelSelect } from "./settings/RobotModelSelect";
import { SpaceMouseModeSelect } from "./settings/SpaceMouseModeSelect";
import type { RobotModelKey } from "../session/robot-model";

export function ZapdosTopOverlay({
  activeRobotModelKey,
  defaultDebugOpen = false,
  defaultSettingsOpen = false,
  mode = "translate",
  onRobotModelChange,
  selectedBody = null,
  sess,
  sse,
}: {
  activeRobotModelKey: RobotModelKey | null
  defaultDebugOpen?: boolean
  defaultSettingsOpen?: boolean
  mode?: "translate" | "rotate"
  onRobotModelChange: (key: RobotModelKey) => void
  selectedBody?: string | null
  sess: string
  sse: number
}) {
  const [configOpen, setConfigOpen] = useState(defaultSettingsOpen);
  const [debugOpen, setDebugOpen] = useState(defaultDebugOpen);
  const overlayButtonClassName = "rounded-md border border-white/20 bg-black/60 px-3 py-2 text-sm text-white backdrop-blur-sm";

  return <>
    <div className="absolute left-8 top-8">
      <div className="rounded-md bg-black/60 px-3 py-2 text-white backdrop-blur-sm">
        SSE { sse.toFixed(2) } Hz
        <div>Selected { selectedBody ?? "none" }</div>
        <div>Mode { mode }</div>
      </div>
    </div>
    <div className="absolute right-8 top-8 flex items-start gap-3">
      <div className="relative">
        <button
          aria-expanded={ configOpen }
          aria-haspopup="dialog"
          aria-label={ configOpen ? "Close config" : "Open config" }
          className={ overlayButtonClassName }
          onClick={ () => {
            setConfigOpen(current => {
              const next = !current;
              if (next) setDebugOpen(false);
              return next;
            });
          } }
          type="button">
          Config
        </button>
        { configOpen ? <div className="absolute right-0 mt-3 flex w-72 max-w-[calc(100vw-4rem)] flex-col gap-3">
          <RobotModelSelect activeRobotModelKey={ activeRobotModelKey } onChange={ onRobotModelChange } />
          <SpaceMouseModeSelect />
          <CameraOverrideSaveButton sess={ sess } />
        </div> : null }
      </div>
      <div className="relative">
        <button
          aria-expanded={ debugOpen }
          aria-haspopup="dialog"
          aria-label={ debugOpen ? "Close debug" : "Open debug" }
          className={ overlayButtonClassName }
          onClick={ () => {
            setDebugOpen(current => {
              const next = !current;
              if (next) setConfigOpen(false);
              return next;
            });
          } }
          type="button">
          Debug
        </button>
        { debugOpen ? <div className="absolute right-0 mt-3 flex w-72 max-w-[calc(100vw-4rem)] flex-col gap-3">
          <AddBenchmarkTableButton sess={ sess } />
          <GrabTheAppleButton sess={ sess } />
          <PlaceTheAppleButton sess={ sess } />
        </div> : null }
      </div>
    </div>
  </>;
}
