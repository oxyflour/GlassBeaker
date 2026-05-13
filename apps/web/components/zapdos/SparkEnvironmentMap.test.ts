import assert from "node:assert/strict";
import test from "node:test";
import { Object3D } from "three";

import {
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
