'use client'

import { Suspense, useEffect, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

import { ZapdosScene } from "../../../components/zapdos/ZapdosScene";
import {
  buildRobotModelHref,
  getRobotModelKeyFromUsd,
  readPersistedRobotModelKey,
  resolveEffectiveRobotUsd,
  writePersistedRobotModelKey,
  type RobotModelKey,
} from "../../../components/zapdos/robot-model";
import {
  buildZapdosInitStreamUrl,
  buildZapdosSessionStorageKey,
  parseZapdosInitEvent,
  type ZapdosInitPhase,
} from "../../../components/zapdos/zapdos-import";
import { useLocalUUID } from "../../../utils/hooks";

function ZapdosStatus({ message }: { message: string }) {
  return <div className="h-full w-full flex justify-center items-center">
    <div>
      {message}
    </div>
  </div>;
}

function ZapdosInitContent() {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const sceneUsd = searchParams.get("scene_usd");
  const urlRobotUsd = searchParams.get("robot_usd");
  const effectiveRobotUsd = resolveEffectiveRobotUsd(urlRobotUsd, readPersistedRobotModelKey());
  const activeRobotModelKey = getRobotModelKeyFromUsd(effectiveRobotUsd);
  const storageKey = buildZapdosSessionStorageKey(sceneUsd, effectiveRobotUsd)
  const sess = useLocalUUID(storageKey);
  const [state, setState] = useState<{ phase: ZapdosInitPhase; message: string }>({
    phase: "loading",
    message: "loading",
  });

  function handleRobotModelChange(next: RobotModelKey) {
    writePersistedRobotModelKey(next);
    router.replace(buildRobotModelHref(pathname, searchParams.toString(), next));
  }

  useEffect(() => {
    setState({ phase: "loading", message: "loading" });
    const sse = new EventSource(buildZapdosInitStreamUrl(sess, sceneUsd, effectiveRobotUsd));
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
  }, [effectiveRobotUsd, sceneUsd, sess]);
  return state.phase === "started"
    ? <ZapdosScene
      activeRobotModelKey={ activeRobotModelKey }
      onRobotModelChange={ handleRobotModelChange }
      onRuntimeError={ message => setState({ phase: "error", message }) }
      sess={ sess } />
    : <ZapdosStatus message={ state.message } />;
}

export default function ZapdosInit() {
  return <Suspense fallback={ <ZapdosStatus message="loading" /> }>
    <ZapdosInitContent />
  </Suspense>;
}
