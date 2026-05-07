import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

test("standalone agent genie sim route file is removed", () => {
  const routeFile = path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    "agent-genie-sim",
    "page.tsx"
  );

  assert.equal(existsSync(routeFile), false);
});
