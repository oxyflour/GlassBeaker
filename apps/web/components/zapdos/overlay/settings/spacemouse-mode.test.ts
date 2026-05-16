import assert from "node:assert/strict";
import test from "node:test";

type SpaceMouseModeModule = typeof import("./spacemouse-mode");

test("deriveSpaceMouseMode prefers explicit backend mode", async () => {
  const { deriveSpaceMouseMode } = await loadModule<SpaceMouseModeModule>("./spacemouse-mode.ts");

  assert.equal(deriveSpaceMouseMode({ running: true, mode: "left", active_arm: "right" }), "left");
});

test("deriveSpaceMouseMode falls back to off when the manager is stopped", async () => {
  const { deriveSpaceMouseMode } = await loadModule<SpaceMouseModeModule>("./spacemouse-mode.ts");

  assert.equal(deriveSpaceMouseMode({ running: false, active_arm: "right" }), "off");
});

test("createSpaceMouseModeRequest posts the selected mode", async () => {
  const { createSpaceMouseModeRequest } = await loadModule<SpaceMouseModeModule>("./spacemouse-mode.ts");

  assert.deepEqual(createSpaceMouseModeRequest("right"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: "right" }),
  });
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}
