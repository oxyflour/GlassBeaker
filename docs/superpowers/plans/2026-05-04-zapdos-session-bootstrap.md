# Zapdos Session Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/python/zapdos/{sess}/init/start` the only bootstrap entrypoint, keep localStorage session reuse, and surface runtime session loss as an error instead of silently rebuilding with default USDs.

**Architecture:** The backend will split bootstrap-only and runtime-only session access so runtime routes cannot create sessions. Runtime access will also evict stale inactive sessions before use. The frontend will keep the existing `sess` reuse behavior, but when runtime SSE or fetches detect a missing or inactive session, the page will show an explicit error state instead of re-bootstraping automatically.

**Tech Stack:** FastAPI, asyncio, Python unittest, Next.js client components, TypeScript node:test

---

### Task 1: Backend Regression Tests

**Files:**
- Modify: `apps/python/tests/test_zapdos_import.py`
- Test: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Write the failing tests**

```python
    def test_require_session_future_rejects_missing_runtime_session(self):
        with self.assertRaises(MODULE.HTTPException) as err:
            MODULE._require_session_future("sess-1")
        self.assertEqual(err.exception.status_code, 409)

    async def test_require_session_future_evicts_inactive_session(self):
        future: asyncio.Future[object] = asyncio.Future()
        future.set_result(SimpleNamespace(is_active=lambda: False))
        MODULE.sessions["sess-1"] = future

        with self.assertRaises(MODULE.HTTPException) as err:
            MODULE._require_session_future("sess-1")
        self.assertEqual(err.exception.status_code, 409)
        self.assertNotIn("sess-1", MODULE.sessions)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_import`
Expected: FAIL because `_require_session_future` does not exist yet.

### Task 2: Backend Bootstrap Split

**Files:**
- Modify: `apps/python/api/zapdos/{session}/{action}.py`
- Test: `apps/python/tests/test_zapdos_import.py`

- [ ] **Step 1: Add runtime-only session lookup**

```python
def _require_session_future(sess: str) -> asyncio.Future[ZapdosSession]:
    future = sessions.get(sess)
    if future is None:
        raise HTTPException(status_code=409, detail="Session not initialized")
    if future.done() and not future.cancelled() and future.exception() is None:
        session = future.result()
        if not session.is_active():
            sessions.pop(sess, None)
            raise HTTPException(status_code=409, detail="Session expired")
    return future
```

- [ ] **Step 2: Restrict bootstrap creation to `init`**

```python
    if action == "init":
        future = _get_or_create_session_future(req, sess)
        return StreamingResponse(...)

    future = _require_session_future(sess)
    session = await _await_session_future(sess, future)
```

- [ ] **Step 3: Run backend tests**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_import`
Expected: PASS

### Task 3: Frontend Runtime Error Handling

**Files:**
- Create: `apps/web/components/zapdos/zapdos-runtime.ts`
- Modify: `apps/web/app/demo/zapdos/page.tsx`
- Test: `apps/web/components/zapdos/zapdos-runtime.test.ts`

- [ ] **Step 1: Write the failing tests**

```typescript
test("getZapdosRuntimeErrorMessage maps session errors to refresh guidance", () => {
  assert.equal(
    getZapdosRuntimeErrorMessage(new Error('{"detail":"Session not initialized"}')),
    "Session disconnected. Refresh to reload scene."
  );
});

test("isZapdosInactivePayload detects inactive session events", () => {
  assert.equal(isZapdosInactivePayload({ inactive: true }), true);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --filter glassbeaker-web test -- zapdos-runtime`
Expected: FAIL because the helper file does not exist yet.

- [ ] **Step 3: Add helper and wire page error state**

```typescript
export function getZapdosRuntimeErrorMessage(error: unknown) { ... }
export function isZapdosInactivePayload(data: unknown) { ... }
```

```typescript
if (isZapdosInactivePayload(payload)) {
  onError(ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE);
  return;
}
```

- [ ] **Step 4: Run frontend tests**

Run: `pnpm --filter glassbeaker-web test -- zapdos-runtime zapdos-import`
Expected: PASS

### Task 4: End-to-End Verification

**Files:**
- Modify: `apps/python/api/zapdos/{session}/{action}.py`
- Modify: `apps/python/tests/test_zapdos_import.py`
- Create: `apps/web/components/zapdos/zapdos-runtime.ts`
- Create: `apps/web/components/zapdos/zapdos-runtime.test.ts`
- Modify: `apps/web/app/demo/zapdos/page.tsx`

- [ ] **Step 1: Run the focused backend suite**

Run: `uv run --project apps/python --python 3.12 python -m unittest apps.python.tests.test_zapdos_import`
Expected: PASS

- [ ] **Step 2: Run the focused web suite**

Run: `pnpm --filter glassbeaker-web test -- zapdos-runtime zapdos-import`
Expected: PASS
