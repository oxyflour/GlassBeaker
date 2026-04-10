import assert from "node:assert/strict";
import test from "node:test";

type TranspileModule = typeof import("./transpile");

test("buildPreviewModule keeps quotes around rewritten dynamic imports", async () => {
  const { buildPreviewModule } = await loadModule<TranspileModule>("./transpile.ts");
  const source = 'export async function loadThree() { return await import("three"); }\n';

  const result = await buildPreviewModule("/entry.tsx", source, { "/entry.tsx": source }, "https://esm.sh", "http://localhost:3000");

  assert.match(result.code, /import\("https:\/\/esm\.sh\/three"\)/);
  assert.doesNotMatch(result.code, /import\(https:\/\/esm\.sh\/three\)/);
});

test("buildPreviewModule reports preview library modules for validation", async () => {
  const { buildPreviewModule } = await loadModule<TranspileModule>("./transpile.ts");
  const source = 'import Antenna from "@glassbeaker-preview/nijika/Antenna";\nexport default Antenna;\n';

  const result = await buildPreviewModule("/entry.tsx", source, { "/entry.tsx": source }, "https://esm.sh", "http://localhost:3000");

  assert.deepEqual(result.validationImports, ["http://localhost:3000/api/preview/module/%40glassbeaker-preview/nijika/Antenna"]);
});

test("buildPreviewModule rewrites manifold-3d to the local preview package route", async () => {
  const { buildPreviewModule } = await loadModule<TranspileModule>("./transpile.ts");
  const source = 'export async function loadManifold() { return await import("manifold-3d/lib/wasm.js"); }\n';

  const result = await buildPreviewModule("/entry.tsx", source, { "/entry.tsx": source }, "https://esm.sh", "http://localhost:3000");

  assert.match(result.code, /import\("http:\/\/localhost:3000\/api\/preview\/package\/manifold-3d\/lib\/wasm\.js"\)/);
  assert.doesNotMatch(result.code, /esm\.sh\/manifold-3d/);
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  const namespace = (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
  return namespace;
}
