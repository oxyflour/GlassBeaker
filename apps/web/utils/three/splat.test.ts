import assert from "node:assert/strict";
import test from "node:test";

import { attachSparkSplat } from "./splat";

test("attachSparkSplat waits for initialization before reporting ready", async () => {
  const root = new FakeGroup();
  const readyStates: boolean[] = [];
  const initialized = deferred<void>();
  const splat = new FakeSplat(initialized.promise);

  const cleanup = attachSparkSplat({
    createSplat: () => splat,
    onReadyChange: (ready) => {
      readyStates.push(ready);
    },
    root,
    url: "/tmp/point_cloud.ply",
  });

  assert.deepEqual(readyStates, [false]);
  assert.deepEqual(root.added, [splat]);

  initialized.resolve();
  await splat.initialized;

  assert.deepEqual(readyStates, [false, true]);

  cleanup();

  assert.deepEqual(root.removed, [splat]);
  assert.equal(splat.disposed, true);
  assert.deepEqual(readyStates, [false, true, false]);
});

test("attachSparkSplat ignores late initialization after cleanup", async () => {
  const root = new FakeGroup();
  const readyStates: boolean[] = [];
  const initialized = deferred<void>();
  const splat = new FakeSplat(initialized.promise);

  const cleanup = attachSparkSplat({
    createSplat: () => splat,
    onReadyChange: (ready) => {
      readyStates.push(ready);
    },
    root,
    url: "/tmp/point_cloud.ply",
  });

  cleanup();
  initialized.resolve();
  await splat.initialized;

  assert.deepEqual(readyStates, [false, false]);
});

class FakeGroup {
  added: unknown[] = [];
  removed: unknown[] = [];

  add(object: unknown) {
    this.added.push(object);
  }

  remove(object: unknown) {
    this.removed.push(object);
  }
}

class FakeSplat {
  disposed = false;

  constructor(readonly initialized: Promise<void>) {}

  dispose() {
    this.disposed = true;
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((innerResolve) => {
    resolve = innerResolve;
  });
  return { promise, resolve };
}
