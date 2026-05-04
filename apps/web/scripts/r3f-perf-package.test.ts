import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import test from "node:test";

test("r3f-perf font wrapper modules do not point Turbopack at binary sourcemaps", async () => {
  const require = createRequire(import.meta.url);
  const packageEntryPath = require.resolve("r3f-perf");
  const packageDir = path.dirname(path.dirname(packageEntryPath));
  const wrapperFiles = [
    path.join(packageDir, "dist", "roboto.woff.js"),
    path.join(packageDir, "dist", "roboto.woff.mjs"),
  ];

  for (const filePath of wrapperFiles) {
    const source = await readFile(filePath, "utf8");
    assert.ok(
      !source.includes("sourceMappingURL="),
      `${path.basename(filePath)} should not reference an external sourcemap`,
    );
  }
});
