import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";

import HomePage from "./page";

test("HomePage keeps Agent CopilotKit and Robotic rendering but removes Agent Genie Sim", () => {
  const html = renderToStaticMarkup(<HomePage />);

  assert.match(html, /Agent CopilotKit/);
  assert.match(html, /\/demo\/agent-cpk/);
  assert.match(html, /Agent Pi Web/);
  assert.match(html, /\/demo\/agent-pi-web/);
  assert.match(html, /Circuit design/);
  assert.match(html, /\/demo\/chinatsu/);
  assert.match(html, /Antenna design/);
  assert.match(html, /\/demo\/nijika/);
  assert.match(html, /Robotic rendering/);
  assert.match(html, /\/demo\/zapdos/);
  assert.doesNotMatch(html, /Agent Genie Sim/);
  assert.doesNotMatch(html, /\/demo\/agent-genie-sim/);
});
