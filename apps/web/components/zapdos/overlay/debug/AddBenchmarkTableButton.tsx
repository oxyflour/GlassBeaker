'use client'

import { useState } from "react";

import { addBenchmarkTable } from "./add-benchmark-table";

export function AddBenchmarkTableButton({
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
      const payload = await addBenchmarkTable(sess, undefined, onSceneRevision);
      setMessage(`Added ${payload.instance_id}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Add benchmark table failed");
    } finally {
      setBusy(false);
    }
  }

  return <div className="rounded-md bg-black/60 px-3 py-2 text-white backdrop-blur-sm">
    <button
      className="rounded border border-white/20 bg-black/40 px-3 py-1 text-sm disabled:opacity-60"
      disabled={ busy || !sess }
      onClick={ () => void handleClick() }>
      { busy ? "Adding..." : "Add benchmark table" }
    </button>
    { message ? <div className="mt-2 max-w-80 text-xs text-emerald-200">{ message }</div> : null }
    { error ? <div className="mt-2 max-w-80 text-xs text-red-200">{ error }</div> : null }
  </div>;
}
