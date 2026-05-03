"use client";

import { Grid, Html, OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";

import { simPointToThree, simQuaternionToThree } from "./scene-math";
import type { SceneData, SceneObject } from "./scene-types";

function getObjectColor(assetId: string) {
  let hash = 0;
  for (const char of assetId) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  return `hsl(${hash % 360} 58% 60%)`;
}

function SceneBox({ object }: { object: SceneObject }) {
  const position = simPointToThree(object.position);
  const quaternion = simQuaternionToThree(object.quaternion);
  const scale = object.size.map((axis) => Math.max(axis, 0.02)) as [number, number, number];
  return (
    <mesh position={ position } quaternion={ quaternion } scale={ scale }>
      <boxGeometry args={ [1, 1, 1] } />
      <meshStandardMaterial color={ getObjectColor(object.assetId) } roughness={ 0.55 } metalness={ 0.08 } />
      <Html center distanceFactor={ 12 }>
        <div
          style={ {
            background: "rgba(15, 23, 42, 0.85)",
            border: "1px solid rgba(148, 163, 184, 0.35)",
            borderRadius: 6,
            color: "#e2e8f0",
            fontSize: 11,
            padding: "2px 6px",
            whiteSpace: "nowrap",
          } }
        >
          { object.assetId }
        </div>
      </Html>
    </mesh>
  );
}

export default function SceneViewer({ scene }: { scene: SceneData | null }) {
  return (
    <div style={ { height: "100%", position: "relative" } }>
      <Canvas camera={ { position: [5, 4, 7], fov: 45 } }>
        <color attach="background" args={ ["#08111a"] } />
        <ambientLight intensity={ 1.3 } />
        <directionalLight castShadow intensity={ 2 } position={ [6, 10, 4] } />
        <directionalLight intensity={ 0.7 } position={ [-4, 5, -6] } />
        <Grid args={ [12, 12] } cellColor="#1e293b" sectionColor="#334155" fadeDistance={ 24 } fadeStrength={ 1.5 } />
        { scene?.objects.map((object) => <SceneBox key={ object.id } object={ object } />) }
        <OrbitControls makeDefault />
      </Canvas>
    </div>
  );
}
