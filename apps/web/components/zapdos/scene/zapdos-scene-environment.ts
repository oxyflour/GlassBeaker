import type { SparkRenderer } from "@sparkjsdev/spark";
import type { Object3D, Scene, Texture, Vector3 } from "three";

export interface SparkEnvironmentMapOptions {
  captureBackground?: Scene["background"];
  captureEnvironment?: Scene["environment"];
  captureScene?: Scene;
  far?: number;
  hideObjects?: Object3D[];
  includeObjects?: Object3D[];
  near?: number;
  size?: number;
  update?: boolean;
  worldCenter: Vector3;
}

export interface SparkEnvironmentCaptureOptions {
  far: number;
  hideObjects: Object3D[];
  near: number;
  scene: Scene;
  size: number;
  update: boolean;
  worldCenter: Vector3;
}

interface SparkEnvironmentMapControllerOptions {
  captureEnvMap?: (options: SparkEnvironmentCaptureOptions) => Promise<Texture>;
  invalidate: () => void;
  scene: Scene;
  spark: Pick<SparkRenderer, "renderEnvMap"> & Object3D;
}

export function createSparkEnvironmentMapController({
  captureEnvMap,
  invalidate,
  scene,
  spark,
}: SparkEnvironmentMapControllerOptions) {
  let activeToken = 0;
  let currentEnvMap: Texture | null = null;
  let disposed = false;

  return {
    async refresh({
      captureBackground,
      captureEnvironment,
      captureScene,
      far = 100,
      hideObjects,
      includeObjects,
      near = 0.1,
      size = 256,
      update = true,
      worldCenter,
    }: SparkEnvironmentMapOptions) {
      const token = ++activeToken;
      const envScene = captureScene ?? scene;
      const appliedBackground = captureBackground !== undefined;
      const appliedEnvironment = captureEnvironment !== undefined;
      const previousBackground = envScene.background;
      const previousEnvironment = envScene.environment;
      if (appliedBackground) {
        envScene.background = captureBackground;
      }
      if (appliedEnvironment) {
        envScene.environment = captureEnvironment;
      }
      const captureOptions: SparkEnvironmentCaptureOptions = {
        far,
        hideObjects: buildCaptureHideObjects(envScene, hideObjects ?? [], includeObjects, [spark]),
        near,
        scene: envScene,
        size,
        update,
        worldCenter,
      };
      let envMap: Texture;
      try {
        envMap = captureEnvMap
          ? await captureEnvMap(captureOptions)
          : await spark.renderEnvMap(captureOptions);
      } finally {
        if (appliedBackground) {
          envScene.background = previousBackground;
        }
        if (appliedEnvironment) {
          envScene.environment = previousEnvironment;
        }
      }
      if (disposed || token !== activeToken) {
        envMap.dispose();
        return;
      }
      if (currentEnvMap && currentEnvMap !== envMap) {
        currentEnvMap.dispose();
      }
      currentEnvMap = envMap;
      scene.environment = envMap;
      invalidate();
    },
    dispose() {
      disposed = true;
      activeToken += 1;
      if (scene.environment === currentEnvMap) {
        scene.environment = null;
      }
      currentEnvMap?.dispose();
      currentEnvMap = null;
    },
  };
}

function buildCaptureHideObjects(
  scene: Scene,
  hideObjects: Object3D[],
  includeObjects?: Object3D[],
  preserveObjects: Object3D[] = [],
) {
  if (!includeObjects?.length) {
    return hideObjects;
  }
  const visible = collectIncludedSubtrees(includeObjects);
  const hidden = new Set(hideObjects);
  collectExcludedRoots(scene, visible, hidden);
  for (const object of preserveObjects) {
    hidden.delete(object);
  }
  return [...hidden];
}

function collectIncludedSubtrees(includeObjects: Object3D[]) {
  const visible = new Set<Object3D>();
  for (const object of includeObjects) {
    object.traverse((node) => {
      visible.add(node);
    });
    let current: Object3D | null = object;
    while (current) {
      visible.add(current);
      current = current.parent;
    }
  }
  return visible;
}

function collectExcludedRoots(root: Object3D, visible: Set<Object3D>, hidden: Set<Object3D>) {
  for (const child of root.children) {
    if (!visible.has(child)) {
      hidden.add(child);
      continue;
    }
    collectExcludedRoots(child, visible, hidden);
  }
}
