'use client'

import { useState } from "react";

import { pickTheApple } from "./pick-the-apple";

export function PickTheAppleButton({ sess }: { sess: string }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function handleClick() {
    setBusy(true);
    setMessage("");
    setError("");
    try {
      const payload = await pickTheApple(sess);
      setMessage(`Picked ${payload.target_body ?? "apple"}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Pick apple failed");
    } finally {
      setBusy(false);
    }
  }

  return <div className="rounded-md bg-black/60 px-3 py-2 text-white backdrop-blur-sm">
    <button
      className="rounded border border-white/20 bg-black/40 px-3 py-1 text-sm disabled:opacity-60"
      disabled={ busy || !sess }
      onClick={ () => void handleClick() }>
      { busy ? "Picking..." : "Pick the apple" }
    </button>
    { message ? <div className="mt-2 max-w-80 text-xs text-emerald-200">{ message }</div> : null }
    { error ? <div className="mt-2 max-w-80 text-xs text-red-200">{ error }</div> : null }
  </div>;
}
