import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";

import { PickTheAppleButton } from "./PickTheAppleButton";

test("PickTheAppleButton disables picking until an object is selected", () => {
  const html = renderToStaticMarkup(<PickTheAppleButton selectedBody={ null } sess="sess-1" />);

  assert.match(html, /Pick selected object/);
  assert.match(html, /disabled=""/);
});

test("PickTheAppleButton enables picking for the selected object", () => {
  const html = renderToStaticMarkup(<PickTheAppleButton selectedBody="Scene_Crate" sess="sess-1" />);

  assert.match(html, /Pick selected object/);
  assert.match(html, /Target Scene_Crate/);
  assert.doesNotMatch(html, /disabled=""/);
});
