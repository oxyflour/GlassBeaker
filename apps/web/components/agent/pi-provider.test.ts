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
