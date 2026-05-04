import assert from "node:assert/strict";
import test from "node:test";

import {
  ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE,
  getZapdosRuntimeErrorMessage,
  isZapdosInactivePayload,
} from "./zapdos-runtime";

test("getZapdosRuntimeErrorMessage maps missing session errors to refresh guidance", () => {
  assert.equal(
    getZapdosRuntimeErrorMessage(new Error('{"detail":"Session not initialized"}')),
    ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE
  );
});

test("getZapdosRuntimeErrorMessage falls back to raw error messages", () => {
  assert.equal(getZapdosRuntimeErrorMessage(new Error("boom")), "boom");
});

test("isZapdosInactivePayload detects inactive session events", () => {
  assert.equal(isZapdosInactivePayload({ inactive: true }), true);
  assert.equal(isZapdosInactivePayload({ inactive: false }), false);
  assert.equal(isZapdosInactivePayload(null), false);
});
