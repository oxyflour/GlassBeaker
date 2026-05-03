import assert from "node:assert/strict";
import test from "node:test";

import { renderToStaticMarkup } from "react-dom/server";

import { OpenInZapdosLink } from "./OpenInZapdosLink";

test("OpenInZapdosLink encodes the USDA path into the Zapdos href", () => {
  const html = renderToStaticMarkup(<OpenInZapdosLink sceneUsdaPath="C:/tmp/my scene.usda" />);
  assert.match(html, /Open in Zapdos/);
  assert.match(html, /\/demo\/zapdos\?scene_usd=C%3A%2Ftmp%2Fmy%20scene\.usda/);
});
