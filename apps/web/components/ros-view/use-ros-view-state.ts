"use client";

import { startTransition, useEffect, useState } from "react";

import { loadRosViewState, type RosViewState, subscribeRosViewState } from "./api";
import type { RosTopic } from "./demo-data";

interface LocalRosViewState {
  connected: boolean;
  lastError: string | null;
  topics: RosTopic[];
}

const FALLBACK_STATE: LocalRosViewState = {
  connected: false,
  lastError: null,
  topics: [],
};

function mapState(next: RosViewState): LocalRosViewState {
  return {
    connected: next.connected,
    lastError: next.last_error,
    topics: next.topics,
  };
}

export function useRosViewState(): LocalRosViewState {
  const [state, setState] = useState<LocalRosViewState>(FALLBACK_STATE);

  useEffect(() => {
    let active = true;
    const apply = (next: RosViewState) => {
      if (!active) return;
      startTransition(() => setState(mapState(next)));
    };
    const handleError = (error: Error) => {
      if (!active) return;
      startTransition(() => setState((current) => ({
        ...current,
        connected: false,
        lastError: error.message,
      })));
    };

    void loadRosViewState().then(apply).catch((error) => {
      handleError(error instanceof Error ? error : new Error("ROS view backend unavailable"));
    });
    const close = subscribeRosViewState(apply, undefined, handleError);

    return () => {
      active = false;
      close();
    };
  }, []);

  return state;
}
