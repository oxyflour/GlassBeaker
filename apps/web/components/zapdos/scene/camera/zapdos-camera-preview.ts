const MAX_VISIBLE_CAMERAS = 3;

export function createGetCameraRequest(): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([]),
  };
}

export async function loadCameraNames(sess: string): Promise<string[]> {
  const response = await fetch(`/python/zapdos/${sess}/call/get_camera`, createGetCameraRequest());
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const payload = await response.json() as Record<string, number[]>;
  return Object.keys(payload).slice(0, MAX_VISIBLE_CAMERAS);
}

export function buildCompositeCameraStreamUrl(sess: string): string {
  return `/python/zapdos/${sess}/multicam/stream`;
}

export function buildCompositeCameraSlices(
  cameras: readonly string[],
  width: number,
  height: number,
): Array<{ camera: string; left: number; width: number; height: number }> {
  if (cameras.length === 0) return [];
  const sliceWidth = Math.floor(width / cameras.length);
  return cameras.map((camera, index) => ({
    camera,
    left: sliceWidth * index,
    width: sliceWidth,
    height,
  }));
}
