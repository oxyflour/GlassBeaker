export interface SaveCameraOverrideResponse {
  ok: boolean;
  saved: number;
  path: string;
}

export function createSaveCameraOverrideRequest(): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([]),
  };
}

export async function saveCameraOverride(sess: string): Promise<string> {
  const response = await fetch(`/python/zapdos/${sess}/call/save_camera_override`, createSaveCameraOverrideRequest());
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const payload = await response.json() as SaveCameraOverrideResponse;
  return `Saved ${payload.saved} camera overrides to ${payload.path}`;
}
