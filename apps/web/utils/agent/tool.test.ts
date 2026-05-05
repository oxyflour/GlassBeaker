import assert from "node:assert/strict";
import test from "node:test";

import { addAssetToSceneToolArgsSchema } from "../../components/zapdos/zapdos-agent-tool-schemas";
import { buildToolParametersFromZod } from "./tool";

test("buildToolParametersFromZod keeps the discriminant required across all variants", () => {
  const parameters = buildToolParametersFromZod(addAssetToSceneToolArgsSchema);
  const placementParameter = parameters.find((parameter) => parameter.name === "placement");

  assert.equal(placementParameter?.type, "object");

  const kindParameter = placementParameter.attributes?.find((parameter) => parameter.name === "kind");

  assert.equal(kindParameter?.type, "string");
  assert.equal(kindParameter?.required, true);
  assert.deepEqual(kindParameter?.enum, ["floor_at_xy", "on_top_of_body", "world_pose"]);
});
