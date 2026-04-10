import assert from "node:assert/strict";
import test from "node:test";

type RequestModule = typeof import("./request");
type SourceModule = typeof import("./source");

test("buildPreviewModuleSource keeps quotes around Antenna dynamic imports", async () => {
  const { resolvePreviewModuleRequest } = await loadModule<RequestModule>("./request.ts");
  const { buildPreviewModuleSource } = await loadModule<SourceModule>("./source.ts");
  const request = resolvePreviewModuleRequest(["@glassbeaker-preview", "nijika", "Antenna"]);

  const code = await buildPreviewModuleSource(request, "http://localhost:3000");

  assert.match(code, /import\("https:\/\/esm\.sh\/three"\)/);
  assert.match(code, /import\("http:\/\/localhost:3000\/api\/preview\/package\/manifold-3d\/lib\/wasm\.js"\)/);
  assert.doesNotMatch(code, /import\(https:\/\/esm\.sh\/three\)/);
  assert.doesNotMatch(code, /esm\.sh\/manifold-3d/);
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  const namespace = (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
  return namespace;
}
