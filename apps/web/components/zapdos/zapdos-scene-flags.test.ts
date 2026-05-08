import assert from "node:assert/strict";
import test from "node:test";

type ZapdosSceneFlagsModule = typeof import("./zapdos-scene-flags");

test("Zapdos scene camera feeds are disabled by default", async () => {
  const { ENABLE_ZAPDOS_CAMERAS } = await loadModule<ZapdosSceneFlagsModule>("./zapdos-scene-flags.ts");

  assert.equal(ENABLE_ZAPDOS_CAMERAS, false);
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}
