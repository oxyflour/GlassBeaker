import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";

import { PlaceTheAppleButton } from "./PlaceTheAppleButton";

test("PlaceTheAppleButton disables placing until an object is selected", () => {
  const html = renderToStaticMarkup(<PlaceTheAppleButton selectedBody={ null } sess="sess-1" />);

  assert.match(html, /Place selected object/);
  assert.match(html, /disabled=""/);
});

test("PlaceTheAppleButton enables placing for the selected object", () => {
  const html = renderToStaticMarkup(<PlaceTheAppleButton selectedBody="Scene_Crate" sess="sess-1" />);

  assert.match(html, /Place selected object/);
  assert.match(html, /Target Scene_Crate/);
  assert.doesNotMatch(html, /disabled=""/);
});
