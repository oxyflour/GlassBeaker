export const ROBOT_MODEL_STORAGE_KEY = "zapdos.robot-model";

export type RobotModelKey = "r1pro" | "moz1";

export const DEFAULT_ROBOT_MODEL_KEY: RobotModelKey = "r1pro";

const ROBOT_USD_BY_KEY: Record<RobotModelKey, string> = {
  moz1: "deps/spirit01_model/USD/Moz1_robot_only.usda",
  r1pro: "deps/galaxea/object/r1pro/r1pro.usda",
};

function coerceRobotModelKey(value: string | null | undefined): RobotModelKey {
  return value === "moz1" ? "moz1" : "r1pro";
}

export function getRobotUsdForModel(key: RobotModelKey) {
  return ROBOT_USD_BY_KEY[key];
}

export function getRobotModelKeyFromUsd(robotUsd: string | null) {
  for (const [key, value] of Object.entries(ROBOT_USD_BY_KEY)) {
    if (value === robotUsd) {
      return key as RobotModelKey;
    }
  }
  return null;
}

export function readPersistedRobotModelKey(
  storage: Pick<Storage, "getItem"> | null = typeof window === "undefined" ? null : window.localStorage
) {
  return coerceRobotModelKey(storage?.getItem(ROBOT_MODEL_STORAGE_KEY));
}

export function writePersistedRobotModelKey(
  key: RobotModelKey,
  storage: Pick<Storage, "setItem"> | null = typeof window === "undefined" ? null : window.localStorage
) {
  storage?.setItem(ROBOT_MODEL_STORAGE_KEY, key);
}

export function resolveEffectiveRobotUsd(urlRobotUsd: string | null, persistedRobotKey: string | null) {
  return urlRobotUsd?.trim() || getRobotUsdForModel(coerceRobotModelKey(persistedRobotKey));
}

export function buildRobotModelHref(pathname: string, search: string, key: RobotModelKey) {
  const nextParams = new URLSearchParams(search);
  nextParams.set("robot_usd", getRobotUsdForModel(key));
  const suffix = nextParams.toString();
  return suffix ? `${pathname}?${suffix}` : pathname;
}
