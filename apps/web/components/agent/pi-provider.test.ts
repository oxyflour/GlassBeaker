import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { PROVIDER } from "./pi-provider";

type ProviderWithApiKeys = typeof PROVIDER & {
  apiKeys?: Record<string, string>;
};

test("Hermes model uses its own provider id and local API server key", () => {
  const hermes = PROVIDER.models?.find((model) => model.id === "hermes");

  assert.equal(hermes?.provider, "hermes");
  assert.equal((PROVIDER as ProviderWithApiKeys).apiKeys?.hermes, "sk-1234");
});

test("desktop starts Hermes API server with the Next origin allowed", async () => {
  const source = await readFile(new URL("../../../desktop/src/main.cjs", import.meta.url), "utf8");

  assert.match(source, /API_SERVER_CORS_ORIGINS/);
  assert.match(source, /http:\/\/localhost:\$\{nextJsPort\}/);
  assert.match(source, /http:\/\/127\.0\.0\.1:\$\{nextJsPort\}/);
});

test("desktop stores embedded Hermes state under the GlassBeaker home", async () => {
  const source = await readFile(new URL("../../../desktop/src/main.cjs", import.meta.url), "utf8");

  assert.match(source, /GLASSBEAKER_HERMES_HOME/);
  assert.match(source, /app\.getPath\(["']home["']\)/);
  assert.match(source, /\.glass-beaker/);
  assert.match(source, /HERMES_HOME:\s*hermesHome/);
});

test("desktop configures embedded Hermes to use the desktop OpenAI-compatible endpoint", async () => {
  const source = await readFile(new URL("../../../desktop/src/main.cjs", import.meta.url), "utf8");

  assert.match(source, /syncHermesProfile/);
  assert.match(source, /path\.join\(hermesHome,\s*["']config\.yaml["']\)/);
  assert.match(source, /provider:\s*custom/);
  assert.match(source, /COPILOTKIT_MODEL/);
  assert.match(source, /OPENAI_BASE_URL/);
  assert.match(source, /HERMES_INFERENCE_PROVIDER:\s*["']custom["']/);
});

test("desktop lets the Hermes API port be configured for the child and rewrite", async () => {
  const desktopSource = await readFile(new URL("../../../desktop/src/main.cjs", import.meta.url), "utf8");
  const nextSource = await readFile(new URL("../../next.config.ts", import.meta.url), "utf8");

  assert.match(desktopSource, /GLASSBEAKER_HERMES_PORT/);
  assert.match(desktopSource, /API_SERVER_PORT:\s*`\$\{hermesPort\}`/);
  assert.match(nextSource, /GLASSBEAKER_HERMES_PORT/);
});

test("desktop does not inherit messaging platform credentials into embedded Hermes", async () => {
  const source = await readFile(new URL("../../../desktop/src/main.cjs", import.meta.url), "utf8");

  assert.match(source, /messagingEnvPrefixes/);
  assert.match(source, /TELEGRAM_/);
  assert.match(source, /DISCORD_/);
  assert.match(source, /SLACK_/);
  assert.match(source, /delete hermesEnv\[key\]/);
});

test("desktop starts Next before waiting for the Python runtime", async () => {
  const source = await readFile(new URL("../../../desktop/src/main.cjs", import.meta.url), "utf8");
  const nextForkIndex = source.indexOf("utilityProcess.fork");
  const runtimeWaitIndex = source.indexOf("assertUrl(`http://127.0.0.1:${pythonPort}/runtime`)");

  assert.notEqual(nextForkIndex, -1);
  assert.notEqual(runtimeWaitIndex, -1);
  assert.ok(nextForkIndex < runtimeWaitIndex);
  assert.match(source, /Promise\.all\(\[/);
  assert.doesNotMatch(source, /API_RUNTIME:\s*apiRuntime/);
});

test("CopilotKit route loads the Python runtime after Next has started", async () => {
  const source = await readFile(new URL("../../app/api/copilotkit/route.ts", import.meta.url), "utf8");

  assert.match(source, /async function getRuntime/);
  assert.match(source, /fetch\(new URL\(["']runtime["'],/);
  assert.match(source, /status:\s*503/);
  assert.doesNotMatch(source, /JSON\.parse\(process\.env\.API_RUNTIME/);
});
