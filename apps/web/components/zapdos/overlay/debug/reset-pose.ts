import {
  createSceneToolRequest,
  runSceneTask,
  type SceneTaskOptions,
} from "../../agent/zapdos-tool-api";

export type ResetPoseResult = {
  ok: boolean;
  reset_bodies: string[];
  scene_revision: string;
};

export function createResetPoseRequest(): RequestInit {
  return createSceneToolRequest([]);
}

export async function resetPose(
  sess: string,
  options?: SceneTaskOptions,
  onSceneRevision?: (revision: string) => void,
) {
  const payload = await runSceneTask<ResetPoseResult>(
    sess,
    "reset_pose",
    createResetPoseRequest(),
    options,
  );
  onSceneRevision?.(payload.scene_revision);
  return payload;
}
