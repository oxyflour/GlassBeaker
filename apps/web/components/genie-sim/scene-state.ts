"use client";

import { useCallback, useState } from "react";

import type { SceneData } from "./scene-types";

export function useSceneState() {
  const [scene, setScene] = useState<SceneData | null>(null);

  const setSceneData = useCallback((data: SceneData) => {
    setScene(data);
  }, []);

  return {
    hasScene: scene !== null,
    scene,
    setSceneData,
  };
}
