"use client";

import { useEffect, useRef, useState } from "react";
import type { SceneData } from "./scene-types";

interface SceneViewerProps {
  scene: SceneData | null;
}

const DEG2RAD = Math.PI / 180;

function buildMesh(
  THREE: typeof import("three"),
  obj: SceneData["objects"][number],
): import("three").Mesh {
  let geo: import("three").BufferGeometry;
  const [w, h, d] = obj.scale;
  switch (obj.type) {
    case "sphere":
      geo = new THREE.SphereGeometry(Math.max(w, 0.001), 24, 16);
      break;
    case "cylinder":
      geo = new THREE.CylinderGeometry(Math.max(w, 0.001), Math.max(d, 0.001), Math.max(h, 0.001), 24);
      break;
    case "cone":
      geo = new THREE.ConeGeometry(Math.max(w, 0.001), Math.max(h, 0.001), 24);
      break;
    case "plane":
      geo = new THREE.PlaneGeometry(Math.max(w, 0.001), Math.max(h, 0.001));
      break;
    default:
      geo = new THREE.BoxGeometry(Math.max(w, 0.001), Math.max(h, 0.001), Math.max(d, 0.001));
  }
  const mat = new THREE.MeshStandardMaterial({
    color: obj.color || "#888888",
    roughness: 0.5,
    metalness: 0.1,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(...obj.position);
  mesh.rotation.set(obj.rotation[0] * DEG2RAD, obj.rotation[1] * DEG2RAD, obj.rotation[2] * DEG2RAD);
  if (obj.type === "sphere") mesh.scale.set(w, w, w);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

export default function SceneViewer({ scene }: SceneViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const ctxRef = useRef<{
    THREE: typeof import("three") | null;
    sceneGroup: import("three").Group | null;
  }>({ THREE: null, sceneGroup: null });

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let cancelled = false;
    let frameId = 0;
    let renderer: import("three").WebGLRenderer | null = null;
    let controls: import("three/examples/jsm/controls/OrbitControls.js").OrbitControls | null = null;
    let resizeObserver: ResizeObserver | null = null;

    async function setup() {
      const [THREE, { OrbitControls }] = await Promise.all([
        import("three"),
        import("three/examples/jsm/controls/OrbitControls.js"),
      ]);
      if (cancelled) return;

      const threeScene = new THREE.Scene();
      threeScene.background = new THREE.Color("#1a1a2e");

      const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
      camera.position.set(4, 3, 6);

      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      host!.appendChild(renderer.domElement);

      const ambient = new THREE.HemisphereLight("#b0c4de", "#2a1a0a", 1.2);
      const key = new THREE.DirectionalLight("#ffffff", 2.0);
      key.position.set(5, 8, 6);
      key.castShadow = true;
      key.shadow.mapSize.set(1024, 1024);
      threeScene.add(ambient, key);

      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.target.set(0, 0.5, 0);

      const group = new THREE.Group();
      threeScene.add(group);
      ctxRef.current.THREE = THREE;
      ctxRef.current.sceneGroup = group;

      const resize = () => {
        if (!renderer) return;
        const w = host?.clientWidth || 1;
        const h = host?.clientHeight || 1;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      };
      resize();
      resizeObserver = new ResizeObserver(resize);
      host && resizeObserver.observe(host);

      const animate = () => {
        if (cancelled) return;
        controls?.update();
        renderer?.render(threeScene, camera);
        frameId = requestAnimationFrame(animate);
      };
      setStatus("ready");
      animate();
    }

    setup().catch(() => setStatus("error"));
    return () => {
      cancelled = true;
      cancelAnimationFrame(frameId);
      resizeObserver?.disconnect();
      controls?.dispose();
      renderer?.dispose();
      renderer?.domElement.remove();
    };
  }, []);

  useEffect(() => {
    const { THREE, sceneGroup } = ctxRef.current;
    if (!THREE || !sceneGroup) return;

    while (sceneGroup.children.length) {
      const child = sceneGroup.children[0] as import("three").Mesh;
      child.geometry.dispose();
      if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
      else child.material.dispose();
      sceneGroup.remove(child);
    }

    if (!scene) return;

    if (scene.groundPlane) {
      const { size, color } = scene.groundPlane;
      const geo = new THREE.PlaneGeometry(size, size);
      const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.8 });
      const ground = new THREE.Mesh(geo, mat);
      ground.rotation.x = -Math.PI / 2;
      ground.receiveShadow = true;
      sceneGroup.add(ground);
    }

    for (const obj of scene.objects) {
      sceneGroup.add(buildMesh(THREE, obj));
    }
  }, [scene]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden" }}>
      <div ref={hostRef} style={{ width: "100%", height: "100%" }} />
      {status !== "ready" && (
        <div style={{ position: "absolute", left: 16, top: 16, color: "#ccc", fontSize: 13 }}>
          {status === "loading" ? "Loading 3D viewer..." : "Failed to initialize 3D viewer"}
        </div>
      )}
      {scene && (
        <div style={{ position: "absolute", right: 12, top: 12, color: "#aaa", fontSize: 11, maxWidth: 200, textAlign: "right" }}>
          {scene.description}
        </div>
      )}
    </div>
  );
}
