import assert from "node:assert/strict";
import test from "node:test";
import { Object3D, Scene, Texture, Vector3 } from "three";

import {
  createSparkEnvironmentMapController,
  type SparkEnvironmentCaptureOptions,
} from "./zapdos-scene-environment";

test("createSparkEnvironmentMapController captures a Spark env map and updates the scene", async () => {
  const scene = new Scene();
  const calls: unknown[] = [];
  const invalidateCalls: number[] = [];
  const envMap = new Texture();
  const disposeCalls: Texture[] = [];
  envMap.dispose = () => {
    disposeCalls.push(envMap);
  };

  const controller = createSparkEnvironmentMapController({
    invalidate: () => {
      invalidateCalls.push(1);
    },
    scene,
    spark: Object.assign(new Object3D(), {
      renderEnvMap: async (options: SparkEnvironmentCaptureOptions) => {
        calls.push(options);
        return envMap;
      },
    }),
  });

  await controller.refresh({
    far: 200,
    hideObjects: [],
    near: 0.25,
    size: 128,
    update: false,
    worldCenter: new Vector3(1, 2, 3),
  });

  assert.equal(scene.environment, envMap);
  assert.equal(invalidateCalls.length, 1);
  assert.deepEqual(calls, [{
    far: 200,
    hideObjects: [],
    near: 0.25,
    scene,
    size: 128,
    update: false,
    worldCenter: new Vector3(1, 2, 3),
  }]);

  controller.dispose();

  assert.equal(scene.environment, null);
  assert.deepEqual(disposeCalls, [envMap]);
});

test("createSparkEnvironmentMapController disposes stale env maps from superseded requests", async () => {
  const scene = new Scene();
  const first = deferred<Texture>();
  const second = deferred<Texture>();
  const firstEnvMap = new Texture();
  const secondEnvMap = new Texture();
  const disposed: Texture[] = [];
  firstEnvMap.dispose = () => {
    disposed.push(firstEnvMap);
  };
  secondEnvMap.dispose = () => {
    disposed.push(secondEnvMap);
  };
  let callCount = 0;

  const controller = createSparkEnvironmentMapController({
    invalidate: () => undefined,
    scene,
    spark: Object.assign(new Object3D(), {
      renderEnvMap: async () => {
        callCount += 1;
        return await (callCount === 1 ? first.promise : second.promise);
      },
    }),
  });

  const firstRefresh = controller.refresh({ hideObjects: [], worldCenter: new Vector3() });
  const secondRefresh = controller.refresh({ hideObjects: [], worldCenter: new Vector3(0, 0, 2) });

  second.resolve(secondEnvMap);
  await secondRefresh;

  assert.equal(scene.environment, secondEnvMap);

  first.resolve(firstEnvMap);
  await firstRefresh;

  assert.equal(scene.environment, secondEnvMap);
  assert.deepEqual(disposed, [firstEnvMap]);

  controller.dispose();

  assert.deepEqual(disposed, [firstEnvMap, secondEnvMap]);
});

test("createSparkEnvironmentMapController converts includeObjects into excluded capture roots", async () => {
  const scene = new Scene();
  const includedParent = new Object3D();
  const includedChild = new Object3D();
  const excludedSibling = new Object3D();
  const excludedRoot = new Object3D();
  includedParent.add(includedChild);
  includedParent.add(excludedSibling);
  scene.add(includedParent);
  scene.add(excludedRoot);
  const calls: Array<{
    hideObjects: Object3D[];
    scene: Scene;
  }> = [];

  const controller = createSparkEnvironmentMapController({
    invalidate: () => undefined,
    scene,
    spark: Object.assign(new Object3D(), {
      renderEnvMap: async (options: SparkEnvironmentCaptureOptions) => {
        calls.push({
          hideObjects: options.hideObjects,
          scene: options.scene,
        });
        return new Texture();
      },
    }),
  });

  await controller.refresh({
    includeObjects: [includedChild],
    worldCenter: new Vector3(),
  });

  assert.equal(calls[0]?.scene, scene);
  assert.deepEqual(calls[0]?.hideObjects, [excludedSibling, excludedRoot]);
});

test("createSparkEnvironmentMapController keeps the Spark renderer visible during includeObjects capture", async () => {
  const scene = new Scene();
  const spark = Object.assign(new Object3D(), {
    renderEnvMap: async (options: { hideObjects: Object3D[]; scene: Scene }) => {
      calls.push({
        hideObjects: options.hideObjects,
        scene: options.scene,
      });
      return new Texture();
    },
  });
  const includedParent = new Object3D();
  const includedChild = new Object3D();
  const excludedRoot = new Object3D();
  const calls: Array<{
    hideObjects: Object3D[];
    scene: Scene;
  }> = [];
  includedParent.add(includedChild);
  scene.add(spark);
  scene.add(includedParent);
  scene.add(excludedRoot);

  const controller = createSparkEnvironmentMapController({
    invalidate: () => undefined,
    scene,
    spark,
  });

  await controller.refresh({
    includeObjects: [includedChild],
    worldCenter: new Vector3(),
  });

  assert.equal(calls[0]?.scene, scene);
  assert.deepEqual(calls[0]?.hideObjects, [excludedRoot]);
});

test("createSparkEnvironmentMapController renders from captureScene when provided", async () => {
  const scene = new Scene();
  const captureScene = new Scene();
  const included = new Object3D();
  const excluded = new Object3D();
  captureScene.add(included);
  captureScene.add(excluded);
  const calls: Array<{
    hideObjects: Object3D[];
    scene: Scene;
  }> = [];

  const controller = createSparkEnvironmentMapController({
    invalidate: () => undefined,
    scene,
    spark: Object.assign(new Object3D(), {
      renderEnvMap: async (options: SparkEnvironmentCaptureOptions) => {
        calls.push({
          hideObjects: options.hideObjects,
          scene: options.scene,
        });
        return new Texture();
      },
    }),
  });

  await controller.refresh({
    captureScene,
    hideObjects: [excluded],
    includeObjects: [included],
    worldCenter: new Vector3(0, 0, 2),
  });

  assert.equal(calls[0]?.scene, captureScene);
  assert.deepEqual(calls[0]?.hideObjects, [excluded]);
});

test("createSparkEnvironmentMapController overlays capture HDR settings only during env rendering", async () => {
  const scene = new Scene();
  const captureScene = new Scene();
  const previousBackground = new Texture();
  const previousEnvironment = new Texture();
  const captureBackground = new Texture();
  const captureEnvironment = new Texture();
  const envMap = new Texture();
  captureScene.background = previousBackground;
  captureScene.environment = previousEnvironment;
  let seenBackground: Scene["background"] | undefined;
  let seenEnvironment: Scene["environment"] | undefined;

  const controller = createSparkEnvironmentMapController({
    invalidate: () => undefined,
    scene,
    spark: Object.assign(new Object3D(), {
      renderEnvMap: async (options: SparkEnvironmentCaptureOptions) => {
        seenBackground = options.scene.background;
        seenEnvironment = options.scene.environment;
        return envMap;
      },
    }),
  });

  await controller.refresh({
    captureBackground,
    captureEnvironment,
    captureScene,
    worldCenter: new Vector3(0, 0, 2),
  });

  assert.equal(seenBackground, captureBackground);
  assert.equal(seenEnvironment, captureEnvironment);
  assert.equal(captureScene.background, previousBackground);
  assert.equal(captureScene.environment, previousEnvironment);
  assert.equal(scene.environment, envMap);
});

test("createSparkEnvironmentMapController prefers a custom captureEnvMap implementation when provided", async () => {
  const scene = new Scene();
  const envMap = new Texture();
  let builtinCalls = 0;
  const customCalls: Array<{ scene: Scene }> = [];

  const controller = createSparkEnvironmentMapController({
    captureEnvMap: async (options) => {
      customCalls.push({ scene: options.scene });
      return envMap;
    },
    invalidate: () => undefined,
    scene,
    spark: Object.assign(new Object3D(), {
      renderEnvMap: async () => {
        builtinCalls += 1;
        return new Texture();
      },
    }),
  });

  await controller.refresh({
    worldCenter: new Vector3(0, 0, 2),
  });

  assert.equal(builtinCalls, 0);
  assert.deepEqual(customCalls, [{ scene }]);
  assert.equal(scene.environment, envMap);
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((innerResolve) => {
    resolve = innerResolve;
  });
  return { promise, resolve };
}
