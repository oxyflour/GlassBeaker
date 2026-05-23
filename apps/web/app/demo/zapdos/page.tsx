'use client'

import { Suspense, useEffect, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

import {
  ZapdosScene,
  buildRobotModelHref,
  buildZapdosSessionStorageKey,
  getRobotModelKeyFromUsd,
  readPersistedRobotModelKey,
  resolveEffectiveRobotUsd,
  runZapdosInitTask,
  type RobotModelKey,
  type ZapdosInitPhase,
  writePersistedRobotModelKey,
} from "../../../components/zapdos";
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
    const controller = new AbortController();
    setState({ phase: "loading", message: "loading" });
    void runZapdosInitTask(
      sess,
      sceneUsd,
      effectiveRobotUsd,
      message => setState({ phase: "loading", message }),
      controller.signal,
    )
      .then(() => setState({ phase: "started", message: "started" }))
      .catch(error => {
        if (controller.signal.aborted) return;
        setState({
          phase: "error",
          message: error instanceof Error ? error.message : "Session bootstrap failed",
        });
      });
    return () => controller.abort();
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
