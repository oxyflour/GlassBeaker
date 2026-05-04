'use client'

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

import { ZapdosScene } from "../../../components/zapdos/ZapdosScene";
import {
  buildZapdosInitStreamUrl,
  buildZapdosSessionStorageKey,
  parseZapdosInitEvent,
  type ZapdosInitPhase,
} from "../../../components/zapdos/zapdos-import";
import { useLocalUUID } from "../../../utils/hooks";

function ZapdosStatus({ message }: { message: string }) {
  return <div className="h-full w-full text-center">{message}</div>;
}

function ZapdosInitContent() {
  const searchParams = useSearchParams();
  const sceneUsd = searchParams.get("scene_usd");
  const robotUsd = searchParams.get("robot_usd");
  const sess = useLocalUUID(buildZapdosSessionStorageKey(sceneUsd, robotUsd));
  const [state, setState] = useState<{ phase: ZapdosInitPhase; message: string }>({
    phase: "loading",
    message: "loading",
  });
  useEffect(() => {
    setState({ phase: "loading", message: "loading" });
    const sse = new EventSource(buildZapdosInitStreamUrl(sess, sceneUsd, robotUsd));
    sse.onmessage = event => {
      const next = parseZapdosInitEvent(event.data);
      setState(next);
      if (next.phase !== "loading") sse.close();
    };
    sse.onerror = () => {
      setState({ phase: "error", message: "Session bootstrap failed" });
      sse.close();
    };
    return () => sse.close();
  }, [robotUsd, sceneUsd, sess]);
  return state.phase === "started"
    ? <ZapdosScene onRuntimeError={ message => setState({ phase: "error", message }) } sess={ sess } />
    : <ZapdosStatus message={ state.message } />;
}

export default function ZapdosInit() {
  return <Suspense fallback={ <ZapdosStatus message="loading" /> }><ZapdosInitContent /></Suspense>;
}
