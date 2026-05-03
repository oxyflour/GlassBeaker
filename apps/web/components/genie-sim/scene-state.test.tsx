import assert from "node:assert/strict";
import test from "node:test";

import { renderToStaticMarkup } from "react-dom/server";

import { useSceneState } from "./scene-state";

function Probe() {
  const state = useSceneState();
  return <pre>{Object.keys(state).sort().join(",")}</pre>;
}

test("scene state only exposes scene data controls", () => {
  const html = renderToStaticMarkup(<Probe />);
  assert.match(html, /hasScene/);
  assert.match(html, /scene/);
  assert.match(html, /setSceneData/);
  assert.doesNotMatch(html, /renderError/);
  assert.doesNotMatch(html, /renderResult/);
  assert.doesNotMatch(html, /renderStatus/);
  assert.doesNotMatch(html, /setRenderFailure/);
  assert.doesNotMatch(html, /setRenderSuccess/);
  assert.doesNotMatch(html, /startRender/);
});
