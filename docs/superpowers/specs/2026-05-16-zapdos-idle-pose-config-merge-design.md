# Zapdos Idle Pose Config Merge Design

**Goal:** Add a repo-owned default config file, merge it with `%USERPROFILE%\.glass-beaker\config.json` at load time, and use `override.position[robot_name]` as the source of a robot-specific idle joint pose that makes Zapdos sessions start from a more neutral whole-body posture.

## Decisions

- Add a new default config file at `apps/desktop/config.json`.
- Keep `%USERPROFILE%\.glass-beaker\config.json` as the user override file.
- Treat `override.position[robot_name]` as a `joint_name -> joint_position` mapping for robot idle pose overrides.
- Make the effective config equal to `deep_merge(default_config, user_config)`, with user values winning on conflicts.
- Keep write paths targeted at the user config file only. Saving camera overrides must not copy repo defaults into the user file.
- Apply the idle pose in the Zapdos physics initialization path, not in the manipulation planner.
- Use a stable robot key such as `r1pro` as the `robot_name` key, not a session id or a USD filename.

## Config Shape

Default config example:

```json
{
  "override": {
    "position": {
      "r1pro": {
        "left_arm_joint1": 0.12,
        "left_arm_joint2": 0.48,
        "left_arm_joint3": -0.18,
        "left_arm_joint4": -1.12,
        "left_arm_joint5": 0.04,
        "left_arm_joint6": 0.72,
        "left_arm_joint7": 0.08,
        "right_arm_joint1": -0.12,
        "right_arm_joint2": -0.48,
        "right_arm_joint3": 0.18,
        "right_arm_joint4": 1.12,
        "right_arm_joint5": -0.04,
        "right_arm_joint6": -0.72,
        "right_arm_joint7": -0.08
      }
    }
  }
}
```

- `override.position` is reserved for robot idle joint poses.
- Each `robot_name` entry is a sparse or complete mapping of MuJoCo joint names to scalar joint positions.
- User config may override only a subset of joints. Missing joints continue to come from the default config.

## Merge Rules

- Add a default-config reader in `apps/python/utils/user_config.py` that resolves `apps/desktop/config.json`.
- Keep a raw user-config reader for code that needs to write back only user-authored data.
- Add one effective-config reader that:
  - reads the default config if present
  - reads the user config if present
  - validates that each root is a JSON object
  - recursively merges nested objects
  - replaces arrays and scalars instead of concatenating them
- Missing config files are allowed and resolve to `{}`.
- Malformed non-object roots remain hard errors.

Example merge:

```json
// default
{
  "override": {
    "position": {
      "r1pro": {
        "left_arm_joint1": 0.12,
        "left_arm_joint2": 0.48
      }
    }
  }
}
```

```json
// user
{
  "override": {
    "position": {
      "r1pro": {
        "left_arm_joint2": 0.52
      }
    }
  }
}
```

```json
// effective
{
  "override": {
    "position": {
      "r1pro": {
        "left_arm_joint1": 0.12,
        "left_arm_joint2": 0.52
      }
    }
  }
}
```

## Idle Pose Application

- Resolve the active robot key during bundle or session setup from the selected robot asset, with `r1pro` as the first supported key.
- Load the merged config before building the initial Zapdos runtime state.
- Apply `override.position[robot_name]` to matching MuJoCo joints during physics initialization, before the first `mj_forward`.
- Ignore unknown joint names in the config only if the robot key is valid but the joint does not exist in the current model. That keeps the config tolerant to small model drift.
- Do not apply idle pose through the manipulation runtime. Pick planning should continue to consume the current robot state after physics initialization.

## Write Paths

- `read_user_config()` should return the merged effective config for readers.
- Add a separate raw user-config loader for write flows such as `save_camera_overrides()`.
- `write_user_config()` continues to write only `%USERPROFILE%\.glass-beaker\config.json`.
- Camera override save keeps its current merge behavior against the raw user config, so unrelated user fields survive and repo defaults are not copied into the user file.

## Error Handling

- If `apps/desktop/config.json` exists but is not a JSON object, raise a runtime error.
- If `%USERPROFILE%\.glass-beaker\config.json` exists but is not a JSON object, raise a runtime error.
- If `USERPROFILE` is unavailable, reading the raw user config for writeback remains an error, but reading the default config alone should still work for pure read paths when possible.
- If `override.position[robot_name]` is not an object, raise a runtime error.
- If a joint position value is not numeric, raise a runtime error instead of silently coercing invalid strings.

## Testing

- Add user-config tests for:
  - missing default file
  - missing user file
  - deep merge of nested objects
  - scalar replacement
  - malformed non-object roots
- Add camera override regression tests showing:
  - merged reads still expose default config values
  - save writes only the user file
  - save does not expand repo defaults into the user file
- Add Zapdos physics or import tests showing:
  - `override.position.r1pro` is applied to initial joint state
  - partial user overrides replace only the specified joints
  - unknown joints do not crash initialization

## Non-Goals

- No planner-level automatic return-to-idle step in this change
- No new frontend editor for idle pose values
- No support for storing body transforms under `override.position`
- No attempt to infer `robot_name` from arbitrary USD filenames beyond the explicitly supported robot keys
