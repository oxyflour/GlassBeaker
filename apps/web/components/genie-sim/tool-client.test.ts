import assert from "node:assert/strict";
import test from "node:test";

import { postToolJson } from "./tool-client";

test("postToolJson returns an error object when fetch rejects", async () => {
  const result = await postToolJson(
    async () => {
      throw new Error("network down");
    },
    "/python/genie_sim/search_assets",
    { query: "table" },
    "Asset search failed"
  );

  assert.deepEqual(result, { error: "Asset search failed: network down" });
});

test("postToolJson returns an error object when a success response has invalid json", async () => {
  const result = await postToolJson(
    async () => ({
      json: async () => {
        throw new Error("bad json");
      },
      ok: true,
      status: 200,
    }),
    "/python/genie_sim/execute",
    { code: "def root_scene(): return []" },
    "Scene execution failed"
  );

  assert.deepEqual(result, { error: "Scene execution failed: bad json" });
});
