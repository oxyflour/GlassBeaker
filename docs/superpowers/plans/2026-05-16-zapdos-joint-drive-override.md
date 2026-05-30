# Zapdos Joint Drive Override Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `override.joint_drive.<robot_key>.<joint_name>` config support so local tuning can override generated MuJoCo joint drive parameters without editing source USD files.

**Architecture:** Parse and validate the config override once during bundle generation, pass the resolved per-joint drive overrides into the USD-to-MJCF converter, and apply them directly to emitted joint and actuator attributes. Keep the override surface narrow: `damping`, `stiffness`, `kp`, and `forcerange`.

**Tech Stack:** Python, MuJoCo, USD, `unittest`

---

### Task 1: Lock the behavior with failing tests

**Files:**
- Modify: `apps/python/tests/test_usd_to_mjcf.py`
- Modify: `apps/python/tests/test_zapdos_idle_pose.py`

- [ ] Add a converter-level test that verifies a hinge joint with a position drive emits overridden `damping`, overridden `actuatorfrcrange`, and overridden actuator `kp`.
- [ ] Run the targeted converter test and confirm it fails because the converter does not accept or apply joint drive overrides yet.
- [ ] Add a config-level test that verifies `override.joint_drive.r1pro.torso_joint1` changes the generated bundle used by MuJoCo startup.
- [ ] Run the targeted config test and confirm it fails because the bundle path does not yet translate config into MJCF drive overrides.

### Task 2: Implement parsing and MJCF override emission

**Files:**
- Add: `apps/python/utils/zapdos/joint_drive_override.py`
- Modify: `apps/python/utils/zapdos/bundle/bundle_builder.py`
- Modify: `apps/python/utils/zapdos/usd_to_mjcf.py`

- [ ] Add a small pure parser/validator for `override.joint_drive`, keyed by robot model key, with strict validation for numbers and `forcerange[2]`.
- [ ] Thread the resolved per-joint overrides through bundle generation into `USDToMJCFConverter`.
- [ ] Apply overrides after USD joint parsing so generated MJCF joint and actuator attributes reflect the config values.
- [ ] Re-run the targeted tests and make them pass with the minimal implementation.

### Task 3: Verify regressions

**Files:**
- Verify only

- [ ] Run focused regression tests around USD-to-MJCF conversion and startup config handling.
- [ ] Check the edited files for accidental scope creep and keep the change set tight.
