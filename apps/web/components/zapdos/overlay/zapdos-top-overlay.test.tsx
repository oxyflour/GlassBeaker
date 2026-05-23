import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";

import { ZapdosTopOverlay } from "./ZapdosTopOverlay";

test("ZapdosTopOverlay keeps SSE on the left and settings collapsed by default", () => {
  const html = renderToStaticMarkup(
    <ZapdosTopOverlay
      activeRobotModelKey="r1pro"
      onRobotModelChange={ () => undefined }
      sess="sess-1"
      sse={12.3456} />
  );

  assert.match(html, /left-8 top-8/);
  assert.match(html, /right-8 top-8/);
  assert.match(html, /SSE 12\.35 Hz/);
  assert.match(html, />Config</);
  assert.match(html, />Debug</);
  assert.doesNotMatch(html, /Add benchmark table/);
  assert.doesNotMatch(html, /Pick the cube/);
  assert.doesNotMatch(html, /Save camera override/);
  assert.doesNotMatch(html, /SpaceMouse/);
});

test("ZapdosTopOverlay keeps Add benchmark table out of the config menu", () => {
  const html = renderToStaticMarkup(
    <ZapdosTopOverlay
      activeRobotModelKey="moz1"
      defaultSettingsOpen
      onRobotModelChange={ () => undefined }
      sess="sess-1"
      sse={1} />
  );

  assert.match(html, /aria-expanded="true"/);
  assert.match(html, /Save camera override/);
  assert.match(html, /Robot model/);
  assert.match(html, /SpaceMouse/);
  assert.match(html, /option value="moz1" selected="">moz1</);
  assert.doesNotMatch(html, /Add benchmark table/);
});

test("ZapdosTopOverlay renders Add benchmark table in the debug menu", () => {
  const html = renderToStaticMarkup(
    <ZapdosTopOverlay
      activeRobotModelKey="moz1"
      defaultDebugOpen
      onRobotModelChange={ () => undefined }
      sess="sess-1"
      sse={1} />
  );

  assert.match(html, /Add benchmark table/);
  assert.match(html, /Reset pose/);
  assert.match(html, /Pick the cube/);
  assert.match(html, /Place the cube/);
  assert.doesNotMatch(html, /Save camera override/);
  assert.doesNotMatch(html, /Robot model/);
  assert.doesNotMatch(html, /SpaceMouse/);
});

test("ZapdosTopOverlay shows the selected body and transform mode", () => {
  const html = renderToStaticMarkup(
    <ZapdosTopOverlay
      activeRobotModelKey="r1pro"
      defaultSettingsOpen={ false }
      onRobotModelChange={ () => undefined }
      sess="sess-1"
      sse={1}
      selectedBody="Scene_Crate"
      mode="rotate" />
  );

  assert.match(html, /Selected Scene_Crate/);
  assert.match(html, /Mode rotate/);
});
