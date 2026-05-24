'use client'

import { useState } from "react";

import { placeSelectedObject } from "./place-the-apple";

export function PlaceTheAppleButton({
  selectedBody,
  sess,
}: {
  selectedBody: string | null;
  sess: string;
}) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const targetLabel = selectedBody?.trim() || "none";

  async function handleClick() {
    if (!selectedBody) {
      setError("Select an object before placing");
      return;
    }
    setBusy(true);
    setMessage("");
    setError("");
    try {
      const payload = await placeSelectedObject(sess, selectedBody);
      setMessage(`Placed ${payload.target_body ?? selectedBody}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Place selected object failed");
    } finally {
      setBusy(false);
    }
  }

  return <div className="rounded-md bg-black/60 px-3 py-2 text-white backdrop-blur-sm">
    <button
      className="rounded border border-white/20 bg-black/40 px-3 py-1 text-sm disabled:opacity-60"
      disabled={ busy || !sess || !selectedBody }
      onClick={ () => void handleClick() }>
      { busy ? "Placing..." : "Place selected object" }
    </button>
    <div className="mt-2 max-w-80 truncate text-xs text-zinc-200">Target { targetLabel }</div>
    { message ? <div className="mt-2 max-w-80 text-xs text-emerald-200">{ message }</div> : null }
    { error ? <div className="mt-2 max-w-80 text-xs text-red-200">{ error }</div> : null }
  </div>;
}
