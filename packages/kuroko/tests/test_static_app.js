const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadApp() {
  const appPath = path.resolve(__dirname, "../static/app.js");
  const plots = [];
  const element = {
    value: "cross",
    files: [],
    addEventListener() {},
  };
  const context = {
    URLSearchParams,
    alert() {},
    document: {
      getElementById() {
        return element;
      },
      querySelectorAll() {
        return [];
      },
    },
    fetch() {
      throw new Error("fetch was not expected in this test");
    },
    Plotly: {
      newPlot(...args) {
        plots.push(args);
      },
      purge() {},
    },
  };

  vm.createContext(context);
  vm.runInContext(readFileSync(appPath, "utf8"), context, { filename: appPath });
  return { context, plots };
}

test("renderPattern3d keeps theta and phi matched in hover text", () => {
  const { context, plots } = loadApp();

  context.renderPattern3d({
    theta: [0, 90, 180],
    phi: [0, 90, 270],
    z: [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ],
  }, "gain");

  const trace = plots[0][1][0];
  const ringStart = 4;

  assert.equal(trace.type, "mesh3d");
  assert.equal(trace.hovertemplate, "%{text}<extra></extra>");
  assert.match(trace.text[ringStart], /phi=0\.0\u00b0/);
  assert.match(trace.text[ringStart], /theta=90\.0\u00b0/);
  assert.match(trace.text[ringStart + 1], /phi=90\.0\u00b0/);
  assert.match(trace.text[ringStart + 1], /theta=90\.0\u00b0/);
  assert.match(trace.text[ringStart + 2], /phi=270\.0\u00b0/);
  assert.match(trace.text[ringStart + 2], /theta=90\.0\u00b0/);
  assert.doesNotMatch(trace.text[ringStart + 2], /theta=270/);
});
