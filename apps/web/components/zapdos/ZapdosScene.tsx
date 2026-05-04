'use client'

import { useEffect, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { Environment, Lightformer, TransformControls } from "@react-three/drei";
import { EffectComposer, N8AO } from "@react-three/postprocessing";
import { Mesh, Object3D, Vector2 } from "three";

import { SparkRendererBridge, SparkSplat } from "../../utils/three/splat";
import { SurfacePivotControls } from "./SurfacePivotControls";
import { ZapdosTopOverlay } from "./ZapdosTopOverlay";
import { buildBodyPosePayload, getSceneVisual, setSceneBodyPose } from "./zapdos-scene-api";
import { applyObjectMatrix, getSceneMaterial, loadSceneGeometry, loadSceneTexture } from "./zapdos-scene-assets";
import { applySceneHotkey, isSelectionClick, pickEditableBodyFromHits, shouldApplyBodyPose, type ZapdosTransformMode } from "./zapdos-scene-state";
import { getZapdosRuntimeErrorMessage, isZapdosInactivePayload, ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE } from "./zapdos-runtime";
import { Perf } from "r3f-perf";
import { Group, Panel } from "react-resizable-panels";
import { CopilotChat } from "@copilotkit/react-core/v2";
import { useGeineSimAssets } from "../../app/demo/agent-genie-sim/page";

const PIVOT_PICK_ROOT = "surface-pivot-content";

class Counter {
  frame = 0;
  start = Date.now();
  flush() {
    const now = Date.now();
    const fps = this.frame / (now - this.start) * 1000;
    this.frame = 0;
    this.start = now;
    return fps;
  }
  record() {
    this.frame += 1;
  }
}

function Cameras({ sess, onRuntimeError }: { sess: string; onRuntimeError: (message: string) => void }) {
  const [cameras, setCameras] = useState([] as string[]);
  useEffect(() => {
    fetch(`/python/zapdos/${sess}/call/get_camera`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([]),
    })
      .then(async response => {
        if (!response.ok) throw new Error(await response.text());
        return await response.json() as Record<string, number[]>;
      })
      .then(result => setCameras(Object.keys(result)))
      .catch(error => onRuntimeError(getZapdosRuntimeErrorMessage(error)));
  }, [onRuntimeError, sess]);
  return <div className="absolute bottom-0 left-0 w-full">{cameras.map(camera => (
    <img
      alt={ camera }
      className="inline"
      key={ camera }
      src={ `/python/zapdos/${sess}/render/${camera}` }
      style={ { width: 320, height: 240, marginLeft: 8, marginBottom: 8 } } />
  ))}</div>;
}

function SceneRuntime({
  mode,
  onRuntimeError,
  sess,
  selectedBody,
  setMode,
  setSelectedBody,
  setSse,
  setTransformDragging,
}: {
  mode: ZapdosTransformMode;
  onRuntimeError: (message: string) => void;
  sess: string;
  selectedBody: string | null;
  setMode: (mode: ZapdosTransformMode) => void;
  setSelectedBody: (body: string | null) => void;
  setSse: (value: number) => void;
  setTransformDragging: (dragging: boolean) => void;
}) {
  const { camera, gl, raycaster, scene } = useThree();
  const bodyObjectsRef = useRef<Record<string, Object3D>>({});
  const draggingBodyRef = useRef<string | null>(null);
  const selectedBodyRef = useRef<string | null>(null);
  const selectionPointerRef = useRef<{ pointerId: number; start: Vector2 } | null>(null);
  selectedBodyRef.current = selectedBody;
  const selectedObject = selectedBody ? bodyObjectsRef.current[selectedBody] ?? null : null;

  useEffect(() => {
    const element = gl.domElement;
    const root = scene.getObjectByName(PIVOT_PICK_ROOT) ?? scene;
    const pointer = new Vector2();
    const pointerAt = (event: PointerEvent) => {
      const rect = element.getBoundingClientRect();
      return new Vector2(event.clientX - rect.left, event.clientY - rect.top);
    };
    const resolveHit = (target: Object3D | null) => {
      let current = target;
      while (current) {
        if ("zapdosEditable" in current.userData || "zapdosBody" in current.userData) {
          return {
            editable: current.userData.zapdosEditable === true,
            body: typeof current.userData.zapdosBody === "string" ? current.userData.zapdosBody as string : null,
          };
        }
        current = current.parent;
      }
      return { editable: false, body: null };
    };
    const onKeyDown = (event: KeyboardEvent) => {
      const next = applySceneHotkey({ mode, selectedBody: selectedBodyRef.current }, event.key);
      if (next.mode !== mode) setMode(next.mode);
      if (next.selectedBody !== selectedBodyRef.current) setSelectedBody(next.selectedBody);
    };
    const onPointerDown = (event: PointerEvent) => {
      if (event.pointerType !== "mouse" || event.button !== 0) return;
      selectionPointerRef.current = { pointerId: event.pointerId, start: pointerAt(event) };
    };
    const onPointerUp = (event: PointerEvent) => {
      if (event.pointerType !== "mouse" || event.button !== 0 || draggingBodyRef.current) return;
      const gesture = selectionPointerRef.current;
      selectionPointerRef.current = null;
      if (!gesture || gesture.pointerId !== event.pointerId) return;
      const end = pointerAt(event);
      if (!isSelectionClick(gesture.start, end)) return;
      pointer.set(end.x / element.clientWidth * 2 - 1, -(end.y / element.clientHeight) * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(root.children, true).map(hit => resolveHit(hit.object));
      setSelectedBody(pickEditableBodyFromHits(hits));
    };
    const clearPointer = (event: PointerEvent) => {
      if (selectionPointerRef.current?.pointerId === event.pointerId) {
        selectionPointerRef.current = null;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    element.addEventListener("pointerdown", onPointerDown);
    element.addEventListener("pointerup", onPointerUp);
    element.addEventListener("pointercancel", clearPointer);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      element.removeEventListener("pointerdown", onPointerDown);
      element.removeEventListener("pointerup", onPointerUp);
      element.removeEventListener("pointercancel", clearPointer);
    };
  }, [camera, gl, mode, raycaster, scene, setMode, setSelectedBody]);

  useEffect(() => {
    const sse = new EventSource(`/python/zapdos/${sess}/call/start`);
    const ping = setInterval(() => void fetch(`/python/zapdos/${sess}/call/ping`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([]),
    }).catch(() => null), 30000);
    const root = scene.getObjectByName(PIVOT_PICK_ROOT) ?? scene;
    const topLevel: Object3D[] = [];
    const counter = new Counter();
    let disposed = false;
    let failed = false;
    const fail = (error: unknown) => {
      if (disposed || failed) return;
      failed = true;
      onRuntimeError(getZapdosRuntimeErrorMessage(error));
      sse.close();
    };
    sse.onmessage = event => {
      const payload = JSON.parse(event.data) as { inactive?: boolean; pose?: Record<string, number[]>; };
      if (isZapdosInactivePayload(payload)) return fail(ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE);
      if (!payload.pose) return;
      for (const [name, matrix] of Object.entries(payload.pose)) {
        if (!shouldApplyBodyPose(name, draggingBodyRef.current)) continue;
        const object = bodyObjectsRef.current[name];
        if (object) applyObjectMatrix(object, matrix);
      }
      counter.record();
      if (counter.frame > 100 || Date.now() - counter.start > 1000) setSse(counter.flush());
    };
    sse.onerror = () => fail(ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE);
    const loadVisuals = async () => {
      const payload = await getSceneVisual(sess);
      for (const body of payload.bodies) {
        const group = new Object3D();
        group.name = body.name;
        group.userData.zapdosBody = body.name;
        group.userData.zapdosEditable = body.editable;
        applyObjectMatrix(group, body.matrix);
        bodyObjectsRef.current[body.name] = group;
        topLevel.push(group);
        root.add(group);
      }
      for (const item of payload.meshes) {
        const geometry = await loadSceneGeometry(item);
        const image = await loadSceneTexture(item.texture);
        const mesh = new Mesh(geometry, getSceneMaterial(item, image));
        mesh.castShadow = !item.name.endsWith(".plane");
        mesh.receiveShadow = true;
        mesh.name = item.name;
        if (item.body) {
          mesh.userData.zapdosBody = item.body;
          mesh.userData.zapdosEditable = bodyObjectsRef.current[item.body]?.userData.zapdosEditable === true;
          applyObjectMatrix(mesh, item.localMatrix as number[]);
          bodyObjectsRef.current[item.body]?.add(mesh);
        } else if (item.matrix) {
          applyObjectMatrix(mesh, item.matrix);
          topLevel.push(mesh);
          root.add(mesh);
        }
      }
    };
    void loadVisuals().catch(fail);
    return () => {
      disposed = true;
      sse.close();
      clearInterval(ping);
      bodyObjectsRef.current = {};
      for (const object of topLevel) root.remove(object);
    };
  }, [onRuntimeError, scene, sess, setSse]);

  const commitSelection = () => {
    const body = selectedBodyRef.current;
    if (!body || !selectedObject) return;
    const pose = buildBodyPosePayload(selectedObject);
    void setSceneBodyPose(sess, body, pose)
      .catch(error => onRuntimeError(getZapdosRuntimeErrorMessage(error)))
      .finally(() => {
        draggingBodyRef.current = null;
        setTransformDragging(false);
      });
  };

  return selectedObject ? <TransformControls
    mode={ mode }
    object={ selectedObject }
    onMouseDown={ () => {
      draggingBodyRef.current = selectedBodyRef.current;
      setTransformDragging(true);
    } }
    onMouseUp={ commitSelection }
    onObjectChange={ () => selectedObject.updateMatrixWorld(true) } /> : null;
}

export function ZapdosScene({ sess, onRuntimeError }: { sess: string; onRuntimeError: (message: string) => void }) {
  const [mode, setMode] = useState<ZapdosTransformMode>("translate");
  const [selectedBody, setSelectedBody] = useState<string | null>(null);
  const [sse, setSse] = useState(0);
  const [transformDragging, setTransformDragging] = useState(false);
  useGeineSimAssets()
  return <Group>
    <Panel className="relative">
    <Canvas camera={ {
        position: [2.5, -2.5, 1.8],
        up: [0, 0, 1],
        fov: 45,
        near: 0.01,
        far: 100,
    } } className="h-full w-full">
      <Perf />
      <SparkRendererBridge />
      <ambientLight intensity={ 1.2 } />
      <directionalLight intensity={ 1.8 } position={ [6, -4, 8] } />
      <directionalLight intensity={ 0.8 } position={ [-4, 6, 4] } />
      <SurfacePivotControls enabled={ !transformDragging } pickRootName={ PIVOT_PICK_ROOT } />
      <EffectComposer multisampling={ 8 }><N8AO aoRadius={ 1 } distanceFalloff={ 1 } intensity={ 4 } /></EffectComposer>
      <Environment resolution={ 256 }><group rotation={ [-Math.PI / 3, 0, 1] }>
        <Lightformer form="circle" intensity={ 4 } position={ [0, 5, -9] } rotation-x={ Math.PI / 2 } scale={ 2 } />
        <Lightformer form="circle" intensity={ 2 } position={ [-5, 1, -1] } rotation-y={ Math.PI / 2 } scale={ 2 } />
        <Lightformer form="circle" intensity={ 2 } position={ [-5, -1, -1] } rotation-y={ Math.PI / 2 } scale={ 2 } />
        <Lightformer form="circle" intensity={ 2 } position={ [10, 1, 0] } rotation-y={ -Math.PI / 2 } scale={ 8 } />
      </group></Environment>
      <group name={ PIVOT_PICK_ROOT }><group position={ [0, 0, 2] }><SparkSplat url="/tmp/butterfly.spz" /></group></group>
      <SceneRuntime
        mode={ mode }
        onRuntimeError={ onRuntimeError }
        selectedBody={ selectedBody }
        sess={ sess }
        setMode={ setMode }
        setSelectedBody={ setSelectedBody }
        setSse={ setSse }
        setTransformDragging={ setTransformDragging } />
    </Canvas>
    <Cameras onRuntimeError={ onRuntimeError } sess={ sess } />
    <ZapdosTopOverlay mode={ mode } selectedBody={ selectedBody } sess={ sess } sse={ sse } />
    </Panel>
    <Panel maxSize="33%">
      <CopilotChat />
    </Panel>
  </Group>
}
