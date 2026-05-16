"use client";

import { useEffect, useMemo, useState } from "react";
import { useThree } from "@react-three/fiber";
import { SparkRenderer } from "@sparkjsdev/spark";
import {
  Camera,
  CubeCamera,
  HalfFloatType,
  LinearMipmapLinearFilter,
  Object3D,
  PMREMGenerator,
  Scene,
  Vector3,
  WebGLCubeRenderTarget,
  WebGLRenderer,
} from "three";

import {
  createSparkEnvironmentMapController,
  type SparkEnvironmentCaptureOptions,
} from "./zapdos-scene-environment";

const EMPTY_OBJECTS: Object3D[] = [];

export function getSceneObjectDependencyKey(objects?: Object3D[]) {
  return objects?.map(object => object.uuid).join(":") ?? "";
}

export function getWorldCenterDependencyKey(worldCenter: readonly [number, number, number]) {
  return worldCenter.join(":");
}

export function SparkEnvironmentMap({
  captureBackground,
  captureEnvironment,
  captureScene,
  far = 100,
  hideObjects,
  includeObjects,
  near = 0.1,
  spark,
  size = 256,
  update = true,
  worldCenter,
}: {
  captureBackground?: Scene["background"];
  captureEnvironment?: Scene["environment"];
  captureScene?: Scene | null;
  far?: number;
  hideObjects?: Object3D[];
  includeObjects?: Object3D[];
  near?: number;
  spark: SparkRenderer | null;
  size?: number;
  update?: boolean;
  worldCenter: readonly [number, number, number];
}) {
  const { gl, invalidate, scene } = useThree();
  const hideObjectsKey = getSceneObjectDependencyKey(hideObjects);
  const includeObjectsKey = getSceneObjectDependencyKey(includeObjects);
  const worldCenterKey = getWorldCenterDependencyKey(worldCenter);
  const stableHideObjects = useMemo(() => hideObjects ?? EMPTY_OBJECTS, [hideObjectsKey]);
  const stableIncludeObjects = useMemo(() => includeObjects, [includeObjectsKey]);
  const stableWorldCenter = useMemo(() => new Vector3(...worldCenter), [worldCenterKey]);
  const [envCapture] = useState(() => createHalfFloatSparkEnvCapture());

  useEffect(() => {
    return () => {
      envCapture.dispose();
    };
  }, [envCapture]);

  const captureEnvMap = useMemo(
    () => spark
      ? (options: SparkEnvironmentCaptureOptions) => envCapture.capture({ ...options, renderer: gl, spark })
      : null,
    [envCapture, gl, spark],
  );

  useEffect(() => {
    if (!spark || !captureEnvMap) {
      return;
    }
    const controller = createSparkEnvironmentMapController({
      captureEnvMap,
      invalidate,
      scene,
      spark,
    });
    void controller.refresh({
      captureBackground,
      captureEnvironment,
      captureScene: captureScene ?? undefined,
      far,
      hideObjects: stableHideObjects,
      includeObjects: stableIncludeObjects,
      near,
      size,
      update,
      worldCenter: stableWorldCenter,
    });
    return () => {
      controller.dispose();
    };
  }, [
    captureBackground,
    captureEnvMap,
    captureEnvironment,
    captureScene,
    far,
    hideObjectsKey,
    includeObjectsKey,
    invalidate,
    near,
    scene,
    size,
    spark,
    stableHideObjects,
    stableIncludeObjects,
    stableWorldCenter,
    update,
    worldCenterKey,
  ]);

  return null;
}

export function createHalfFloatSparkEnvCapture() {
  let cubeRender:
    | {
        camera: CubeCamera;
        far: number;
        near: number;
        target: WebGLCubeRenderTarget;
      }
    | null = null;
  let pmrem: PMREMGenerator | null = null;
  let pmremRenderer: WebGLRenderer | null = null;

  return {
    async capture({
      far,
      hideObjects,
      near,
      renderer,
      scene,
      size,
      spark,
      update,
      worldCenter,
    }: SparkEnvironmentCaptureOptions & {
      renderer: WebGLRenderer;
      spark: Pick<SparkRenderer, "autoUpdate" | "encodeLinear" | "render" | "update">;
    }) {
      if (!cubeRender || cubeRender.target.width !== size || cubeRender.near !== near || cubeRender.far !== far) {
        cubeRender?.target.dispose();
        const target = new WebGLCubeRenderTarget(size, {
          generateMipmaps: true,
          minFilter: LinearMipmapLinearFilter,
        });
        target.texture.type = HalfFloatType;
        cubeRender = { camera: new CubeCamera(near, far, target), far, near, target };
      }
      if (!pmrem || pmremRenderer !== renderer) {
        pmrem?.dispose();
        pmrem = new PMREMGenerator(renderer);
        pmremRenderer = renderer;
      }
      const { camera, target } = cubeRender;
      camera.position.copy(worldCenter);
      camera.updateMatrixWorld();
      if (camera.coordinateSystem !== renderer.coordinateSystem) {
        camera.coordinateSystem = renderer.coordinateSystem;
        camera.updateCoordinateSystem();
      }
      const previousTarget = renderer.getRenderTarget();
      const previousCubeFace = renderer.getActiveCubeFace();
      const previousMipmapLevel = renderer.getActiveMipmapLevel();
      const previousXrEnabled = renderer.xr.enabled;
      const previousAutoClear = renderer.autoClear;
      const generateMipmaps = target.texture.generateMipmaps;
      try {
        await captureSparkSceneForEnvironmentMap({
          hideObjects,
          render: async () => {
            target.texture.generateMipmaps = false;
            renderer.xr.enabled = false;
            renderer.autoClear = true;
            const [cameraPX, cameraNX, cameraPY, cameraNY, cameraPZ, cameraNZ] = camera.children as Camera[];
            renderer.setRenderTarget(target, 0, camera.activeMipmapLevel);
            spark.render(scene, cameraPX);
            renderer.setRenderTarget(target, 1, camera.activeMipmapLevel);
            spark.render(scene, cameraNX);
            renderer.setRenderTarget(target, 2, camera.activeMipmapLevel);
            spark.render(scene, cameraPY);
            renderer.setRenderTarget(target, 3, camera.activeMipmapLevel);
            spark.render(scene, cameraNY);
            renderer.setRenderTarget(target, 4, camera.activeMipmapLevel);
            spark.render(scene, cameraPZ);
            target.texture.generateMipmaps = generateMipmaps;
            renderer.setRenderTarget(target, 5, camera.activeMipmapLevel);
            spark.render(scene, cameraNZ);
          },
          spark,
          update: update ? async () => {
            const tempCamera = new Camera();
            tempCamera.position.copy(worldCenter);
            tempCamera.updateMatrixWorld(true);
            await spark.update({ scene, camera: tempCamera });
          } : undefined,
        });
      } finally {
        target.texture.generateMipmaps = generateMipmaps;
        renderer.setRenderTarget(previousTarget, previousCubeFace, previousMipmapLevel);
        renderer.xr.enabled = previousXrEnabled;
        renderer.autoClear = previousAutoClear;
      }
      target.texture.needsPMREMUpdate = true;
      return pmrem.fromCubemap(target.texture).texture;
    },
    dispose() {
      cubeRender?.target.dispose();
      cubeRender = null;
      pmrem?.dispose();
      pmrem = null;
      pmremRenderer = null;
    },
  };
}

export async function captureSparkSceneForEnvironmentMap({
  hideObjects,
  render,
  spark,
  update,
}: {
  hideObjects: Object3D[];
  render: () => Promise<void>;
  spark: Pick<SparkRenderer, "autoUpdate" | "encodeLinear">;
  update?: () => Promise<void>;
}) {
  if (update) {
    await withSparkCaptureState(spark, update);
  }
  const objectVisibility = new Map<Object3D, boolean>();
  for (const object of hideObjects) {
    objectVisibility.set(object, object.visible);
    object.visible = false;
  }
  try {
    await withSparkCaptureState(spark, render);
  } finally {
    for (const [object, visible] of objectVisibility.entries()) {
      object.visible = visible;
    }
  }
}

export async function withSparkCaptureState<T>(
  spark: Pick<SparkRenderer, "autoUpdate" | "encodeLinear">,
  callback: () => Promise<T>,
) {
  const previousAutoUpdate = spark.autoUpdate;
  const previousEncodeLinear = spark.encodeLinear;
  spark.autoUpdate = false;
  spark.encodeLinear = true;
  try {
    return await callback();
  } finally {
    spark.autoUpdate = previousAutoUpdate;
    spark.encodeLinear = previousEncodeLinear;
  }
}
