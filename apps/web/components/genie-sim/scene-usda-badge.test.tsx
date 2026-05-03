import assert from "node:assert/strict";
import test from "node:test";

import { renderToStaticMarkup } from "react-dom/server";

import { SceneUsdaBadge } from "./scene-usda-badge";

test("usda path is rendered in a floating badge", () => {
  const html = renderToStaticMarkup(<SceneUsdaBadge path="C:/tmp/scene.usda" />);
  assert.match(html, /data-slot="usda-badge"/);
  assert.match(html, /scene\.usda/);
  assert.match(html, /pointer-events-none/);
});
