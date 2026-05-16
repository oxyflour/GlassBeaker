import assert from "node:assert/strict";
import test from "node:test";
import { Object3D } from "three";

import {
  captureSparkSceneForEnvironmentMap,
  getSceneObjectDependencyKey,
  getWorldCenterDependencyKey,
  withSparkCaptureState,
} from "./SparkEnvironmentMap";

test("getSceneObjectDependencyKey stays stable for new arrays with the same objects", () => {
  const first = new Object3D();
  const second = new Object3D();

  assert.equal(
    getSceneObjectDependencyKey([first, second]),
    getSceneObjectDependencyKey([first, second]),
  );
  assert.notEqual(
    getSceneObjectDependencyKey([first, second]),
    getSceneObjectDependencyKey([second, first]),
  );
});

test("getWorldCenterDependencyKey depends on tuple values rather than array identity", () => {
  assert.equal(
    getWorldCenterDependencyKey([0, 0, 2]),
    getWorldCenterDependencyKey([0, 0, 2]),
  );
  assert.notEqual(
    getWorldCenterDependencyKey([0, 0, 2]),
    getWorldCenterDependencyKey([0, 0, 3]),
  );
});

test("withSparkCaptureState enables linear capture and disables auto updates only during the callback", async () => {
  const spark = { autoUpdate: true, encodeLinear: false };
  const seen: Array<{ autoUpdate: boolean; encodeLinear: boolean }> = [];

  await withSparkCaptureState(spark, async () => {
    seen.push({ autoUpdate: spark.autoUpdate, encodeLinear: spark.encodeLinear });
  });

  assert.deepEqual(seen, [{ autoUpdate: false, encodeLinear: true }]);
  assert.deepEqual(spark, { autoUpdate: true, encodeLinear: false });
});

test("withSparkCaptureState restores spark flags after errors", async () => {
  const spark = { autoUpdate: true, encodeLinear: false };

  await assert.rejects(
    () => withSparkCaptureState(spark, async () => {
      throw new Error("boom");
    }),
    /boom/,
  );

  assert.deepEqual(spark, { autoUpdate: true, encodeLinear: false });
});

test("captureSparkSceneForEnvironmentMap keeps scene objects visible while spark updates", async () => {
  const hidden = new Object3D();
  const spark = { autoUpdate: true, encodeLinear: false };
  const seen: Array<{
    autoUpdate: boolean;
    encodeLinear: boolean;
    phase: string;
    visible: boolean;
  }> = [];
  const gate = deferred<void>();

  const capture = captureSparkSceneForEnvironmentMap({
    hideObjects: [hidden],
    render: async () => {
      seen.push({
        autoUpdate: spark.autoUpdate,
        encodeLinear: spark.encodeLinear,
        phase: "render",
        visible: hidden.visible,
      });
    },
    spark,
    update: async () => {
      seen.push({
        autoUpdate: spark.autoUpdate,
        encodeLinear: spark.encodeLinear,
        phase: "update",
        visible: hidden.visible,
      });
      await gate.promise;
    },
  });

  await Promise.resolve();

  assert.equal(hidden.visible, true);
  gate.resolve();
  await capture;

  assert.deepEqual(seen, [
    { autoUpdate: false, encodeLinear: true, phase: "update", visible: true },
    { autoUpdate: false, encodeLinear: true, phase: "render", visible: false },
  ]);
  assert.equal(hidden.visible, true);
  assert.deepEqual(spark, { autoUpdate: true, encodeLinear: false });
});

test("captureSparkSceneForEnvironmentMap restores scene visibility after render errors", async () => {
  const hidden = new Object3D();
  const spark = { autoUpdate: true, encodeLinear: false };

  await assert.rejects(
    () => captureSparkSceneForEnvironmentMap({
      hideObjects: [hidden],
      render: async () => {
        assert.equal(hidden.visible, false);
        throw new Error("boom");
      },
      spark,
    }),
    /boom/,
  );

  assert.equal(hidden.visible, true);
  assert.deepEqual(spark, { autoUpdate: true, encodeLinear: false });
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((innerResolve) => {
    resolve = innerResolve;
  });
  return { promise, resolve };
}
