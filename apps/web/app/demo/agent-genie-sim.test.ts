import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

function getBuiltAppRoutes(): string[] {
  const manifestFile = path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    "..",
    "..",
    ".next",
    "server",
    "app-paths-manifest.json"
  );

  if (!existsSync(manifestFile)) {
    return [];
  }

  const manifest = JSON.parse(readFileSync(manifestFile, "utf8")) as Record<string, string>;
  return Object.keys(manifest);
}

test("standalone agent genie sim route file is removed", () => {
  const routeFile = path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    "agent-genie-sim",
    "page.tsx"
  );

  assert.equal(existsSync(routeFile), false);

  const builtRoutes = getBuiltAppRoutes();
  assert.equal(builtRoutes.includes("/demo/agent-genie-sim"), false);
  assert.equal(builtRoutes.includes("/demo/agent-genie-sim/page"), false);
});
