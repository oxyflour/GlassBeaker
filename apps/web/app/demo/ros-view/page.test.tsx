import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";

import RosViewDemoPage from "./page";

test("RosViewDemoPage renders a FoxGlove-style ROS workbench", () => {
  const html = renderToStaticMarkup(<RosViewDemoPage />);

  assert.match(html, /Split Right/);
  assert.match(html, /Split Down/);
  assert.match(html, /No ROS topics/);
  assert.doesNotMatch(html, /ROS Mission Control/);
});
