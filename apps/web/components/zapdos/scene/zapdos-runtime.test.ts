import assert from "node:assert/strict";
import test from "node:test";

import {
  ZAPDOS_SCENE_REVISION_EVENT,
  ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE,
  createZapdosSceneRevisionEvent,
  getZapdosSceneRevisionEventDetail,
  getZapdosRuntimeErrorMessage,
  getZapdosSceneRevision,
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

test("getZapdosRuntimeErrorMessage keeps IsaacSim log guidance intact", () => {
  assert.equal(
    getZapdosRuntimeErrorMessage(new Error("IsaacSim quit unexpectedly, check C:/tmp/renderer.log")),
    "IsaacSim quit unexpectedly, check C:/tmp/renderer.log"
  );
});

test("isZapdosInactivePayload detects inactive session events", () => {
  assert.equal(isZapdosInactivePayload({ inactive: true }), true);
  assert.equal(isZapdosInactivePayload({ inactive: false }), false);
  assert.equal(isZapdosInactivePayload(null), false);
});

test("getZapdosSceneRevision returns the revision string when present", () => {
  assert.equal(getZapdosSceneRevision({ scene_revision: "rev-2" }), "rev-2");
  assert.equal(getZapdosSceneRevision({ pose: {} }), null);
});

test("createZapdosSceneRevisionEvent carries the session scoped scene revision", () => {
  const event = createZapdosSceneRevisionEvent("sess-1", "rev-2", { force: true });

  assert.equal(event.type, ZAPDOS_SCENE_REVISION_EVENT);
  assert.deepEqual(getZapdosSceneRevisionEventDetail(event, "sess-1"), {
    sess: "sess-1",
    scene_revision: "rev-2",
    force: true,
  });
});

test("getZapdosSceneRevisionEventDetail ignores other sessions", () => {
  const event = createZapdosSceneRevisionEvent("sess-2", "rev-2");

  assert.equal(getZapdosSceneRevisionEventDetail(event, "sess-1"), null);
});
