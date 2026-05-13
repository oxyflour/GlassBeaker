import assert from "node:assert/strict";
import test from "node:test";

import { setSceneAssetsToolArgsSchema } from "../../components/zapdos/zapdos-agent-tool-schemas";
import { pickObjectToolArgsSchema } from "../../components/zapdos/zapdos-manipulation-tool-schemas";
import { buildToolParametersFromZod, formatToolError } from "./tool";

test("buildToolParametersFromZod exposes set_scene_assets assets as an array parameter", () => {
  const parameters = buildToolParametersFromZod(setSceneAssetsToolArgsSchema);
  const assetsParameter = parameters.find((parameter) => parameter.name === "assets") as {
    description?: string;
    required?: boolean;
    type?: string;
  } | undefined;

  assert.equal(assetsParameter?.type, "object[]");
  assert.equal(assetsParameter?.required, true);
  assert.match(assetsParameter?.description ?? "", /asset_id/i);
  assert.match(assetsParameter?.description ?? "", /on_top_of_body/);
});

test("formatToolError keeps the real error message", () => {
  assert.equal(formatToolError(new Error("Scene rebuild already in progress")), "Scene rebuild already in progress");
});

test("formatToolError falls back to stringifying non-Error values", () => {
  assert.equal(formatToolError({ detail: "bad request" }), '[object Object]');
});

test("buildToolParametersFromZod exposes pick_object fields and optional support query", () => {
  const parameters = buildToolParametersFromZod(pickObjectToolArgsSchema);
  const targetQuery = parameters.find((parameter) => parameter.name === "target_query");
  const supportQuery = parameters.find((parameter) => parameter.name === "support_query");
  const arm = parameters.find((parameter) => parameter.name === "arm");

  assert.equal(targetQuery?.type, "string");
  assert.equal(targetQuery?.required, true);
  assert.equal(supportQuery?.type, "string");
  assert.equal(supportQuery?.required, false);
  assert.equal(arm?.type, "string");
  assert.equal(arm?.required, false);
  assert.deepEqual(arm?.enum, ["left", "right"]);
});
