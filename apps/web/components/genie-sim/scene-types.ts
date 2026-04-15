export interface SceneObject {
  id: string;
  type: "box" | "cylinder" | "sphere" | "plane" | "cone";
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];
  color: string;
  label: string;
}

export interface SceneData {
  objects: SceneObject[];
  groundPlane?: { size: number; color: string };
  description: string;
}
