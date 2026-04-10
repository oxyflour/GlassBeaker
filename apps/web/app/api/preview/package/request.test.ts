import assert from "node:assert/strict";
import test from "node:test";

type RequestModule = typeof import("./request");

test("resolvePreviewPackageRequest resolves whitelisted manifold files", async () => {
  const { resolvePreviewPackageRequest } = await loadModule<RequestModule>("./request.ts");

  const resolved = resolvePreviewPackageRequest(["manifold-3d", "lib", "wasm.js"]);

  assert.equal(resolved.packageName, "manifold-3d");
  assert.equal(resolved.relativePath, "lib/wasm.js");
  assert.match(resolved.filePath, /manifold-3d[\\/].*lib[\\/]wasm\.js$/);
  assert.equal(resolved.contentType, "application/javascript; charset=utf-8");
});

test("resolvePreviewPackageRequest rejects non-whitelisted manifold files", async () => {
  const { PreviewPackageError, resolvePreviewPackageRequest } = await loadModule<RequestModule>("./request.ts");

  assert.throws(
    () => resolvePreviewPackageRequest(["manifold-3d", "lib", "worker.js"]),
    (error) => error instanceof PreviewPackageError && /not whitelisted/.test(error.message),
  );
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  const namespace = (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
  return namespace;
}
