import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";

import { ZapdosTopOverlay } from "./ZapdosTopOverlay";

test("ZapdosTopOverlay keeps SSE on the left and settings collapsed by default", () => {
  const html = renderToStaticMarkup(<ZapdosTopOverlay sess="sess-1" sse={12.3456} />);

  assert.match(html, /left-8 top-8/);
  assert.match(html, /right-8 top-8/);
  assert.match(html, /SSE 12\.35 Hz/);
  assert.match(html, />Config</);
  assert.doesNotMatch(html, /Save camera override/);
  assert.doesNotMatch(html, /SpaceMouse/);
});

test("ZapdosTopOverlay renders SpaceMouse and camera save controls when settings are open", () => {
  const html = renderToStaticMarkup(<ZapdosTopOverlay defaultSettingsOpen sess="sess-1" sse={1} />);

  assert.match(html, /aria-expanded="true"/);
  assert.match(html, /Save camera override/);
  assert.match(html, /SpaceMouse/);
});
