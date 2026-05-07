import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

function getManifestFile(): string {
  return path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    "..",
    "..",
    ".next",
    "server",
    "app-paths-manifest.json"
  );
}

function assertBuiltRouteRemoved() {
  const manifestFile = getManifestFile();

  assert.equal(
    existsSync(manifestFile),
    true,
    "Expected Next build output manifest when REQUIRE_NEXT_BUILD_OUTPUT=1"
  );

  const manifest = JSON.parse(readFileSync(manifestFile, "utf8")) as Record<string, string>;
  const builtRoutes = Object.keys(manifest);

  assert.equal(builtRoutes.includes("/demo/agent-genie-sim"), false);
  assert.equal(builtRoutes.includes("/demo/agent-genie-sim/page"), false);
}

test("standalone agent genie sim route file is removed", () => {
  const routeFile = path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    "agent-genie-sim",
    "page.tsx"
  );

  assert.equal(existsSync(routeFile), false);

  if (process.env.REQUIRE_NEXT_BUILD_OUTPUT === "1") {
    assertBuiltRouteRemoved();
  }
});
