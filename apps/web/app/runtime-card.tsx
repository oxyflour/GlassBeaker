"use client";

import { useEffect, useState } from "react";

const fallbackRuntime: GlassBeakerRuntime = {
  isElectron: false,
  electron: "browser",
  chrome: "browser",
  node: "browser",
  packaged: false
};

export default function RuntimeCard() {
  const [runtime, setRuntime] = useState<GlassBeakerRuntime>(fallbackRuntime);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    if (window.glassBeaker) {
      setRuntime(window.glassBeaker);
    }
  }, []);

  return (
    <section className="runtime-card">
      <div className="runtime-pill">
        {runtime.isElectron ? "Running inside Electron" : "Running in browser mode"}
      </div>
      <dl className="runtime-grid">
        <div>
          <dt>Electron</dt>
          <dd>{runtime.electron}</dd>
        </div>
        <div>
          <dt>Chrome</dt>
          <dd>{runtime.chrome}</dd>
        </div>
        <div>
          <dt>Node</dt>
          <dd>{runtime.node}</dd>
        </div>
        <div>
          <dt>Package Mode</dt>
          <dd>{runtime.packaged ? "packaged" : "development"}</dd>
        </div>
      </dl>
    </section>
  );
}
