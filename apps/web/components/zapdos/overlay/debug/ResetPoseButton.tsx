'use client'

import { useState } from "react";

import { resetPose } from "./reset-pose";

export function ResetPoseButton({
  onSceneRevision,
  sess,
}: {
  onSceneRevision?: (revision: string) => void;
  sess: string;
}) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function handleClick() {
    setBusy(true);
    setMessage("");
    setError("");
    try {
      const payload = await resetPose(sess, undefined, onSceneRevision);
      const count = payload.reset_bodies.length;
      setMessage(count ? `Reset ${count} ${count === 1 ? "pose" : "poses"}` : "Pose already reset");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Reset pose failed");
    } finally {
      setBusy(false);
    }
  }

  return <div className="rounded-md bg-black/60 px-3 py-2 text-white backdrop-blur-sm">
    <button
      className="rounded border border-white/20 bg-black/40 px-3 py-1 text-sm disabled:opacity-60"
      disabled={ busy || !sess }
      onClick={ () => void handleClick() }>
      { busy ? "Resetting..." : "Reset pose" }
    </button>
    { message ? <div className="mt-2 max-w-80 text-xs text-emerald-200">{ message }</div> : null }
    { error ? <div className="mt-2 max-w-80 text-xs text-red-200">{ error }</div> : null }
  </div>;
}
