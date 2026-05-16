'use client'

import { type Dispatch, type SetStateAction, useEffect, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { Environment, TransformControls, useEnvironment } from "@react-three/drei";
import { EffectComposer, N8AO } from "@react-three/postprocessing";
import { SparkRenderer } from "@sparkjsdev/spark";
import { Mesh, Object3D, Vector2 } from "three";

import { SparkRendererBridge, SparkSplat } from "../../utils/three/splat";
import { SparkEnvironmentMap } from "./SparkEnvironmentMap";
import { SurfacePivotControls } from "./SurfacePivotControls";
import { ZapdosCameraStrip } from "./ZapdosCameraStrip";
import { ZapdosTopOverlay } from "./ZapdosTopOverlay";
import { buildBodyPosePayload, getSceneVisual, setSceneBodyPose } from "./zapdos-scene-api";
import { applyObjectMatrix, getSceneMaterial, loadSceneMeshResources } from "./zapdos-scene-assets";
import { applySceneHotkey, clearMissingSelection, getDraggedBodyMatrices, getTransformBodyName, isSelectionClick, pickSelectableBodyFromHits, shouldApplyBodyPose, shouldReloadSceneRevision, type ZapdosBodyState, type ZapdosTransformMode } from "./zapdos-scene-state";
import { getZapdosRuntimeErrorMessage, getZapdosSceneRevision, isZapdosInactivePayload, ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE } from "./zapdos-runtime";
import { useZapdosAgentTools } from "./useZapdosAgentTools";
import type { RobotModelKey } from "./robot-model";
import { Perf } from "r3f-perf";
import { Group, Panel } from "react-resizable-panels";
import { CopilotChat } from "@copilotkit/react-core/v2";

const PIVOT_PICK_ROOT = "surface-pivot-content";
const SPARK_ENV_WORLD_CENTER = [0, 0, 2] as const;
const SPARK_ENV_HDR = "/studio_small_03_1k.hdr";

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

function SceneEnvironment({ spark, splatReady, splatRoot }: {
  spark: SparkRenderer | null;
  splatRoot: Object3D | null;
  splatReady: boolean;
}) {
  const hdr = useEnvironment({ files: SPARK_ENV_HDR });
  return <>
    <Environment map={ hdr } />
    {spark && splatRoot && splatReady ? <SparkEnvironmentMap
      captureBackground={ hdr }
      captureEnvironment={ hdr }
      includeObjects={ [splatRoot] }
      spark={ spark }
      worldCenter={ SPARK_ENV_WORLD_CENTER } /> : null}
  </>;
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
  setSelectedBody: Dispatch<SetStateAction<string | null>>;
  setSse: (value: number) => void;
  setTransformDragging: (dragging: boolean) => void;
}) {
  const { camera, gl, raycaster, scene } = useThree();
  const bodyObjectsRef = useRef<Record<string, Object3D>>({});
  const bodyStateRef = useRef<Record<string, ZapdosBodyState>>({});
  const dragPreviewBodiesRef = useRef<Record<string, ZapdosBodyState> | null>(null);
  const draggingBodyRef = useRef<string | null>(null);
  const sceneRevisionRef = useRef<string | null>(null);
  const selectedBodyRef = useRef<string | null>(null);
  const selectionPointerRef = useRef<{ pointerId: number; start: Vector2 } | null>(null);
  selectedBodyRef.current = selectedBody;
  const transformBody = getTransformBodyName(selectedBody, bodyStateRef.current);
  const selectedObject = transformBody ? bodyObjectsRef.current[transformBody] ?? null : null;
  const selectedMovableObject = selectedObject?.userData.zapdosMovable === true ? selectedObject : null;

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
            selectionBody: typeof current.userData.zapdosSelectionBody === "string"
              ? current.userData.zapdosSelectionBody as string
              : null,
          };
        }
        current = current.parent;
      }
      return { editable: false, body: null, selectionBody: null };
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
      setSelectedBody(pickSelectableBodyFromHits(hits));
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
    const sse = new EventSource(`/python/zapdos/${sess}/stream/start`);
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
    const clearLoadedVisuals = () => {
      bodyObjectsRef.current = {};
      bodyStateRef.current = {};
      for (const object of topLevel) {
        root.remove(object);
      }
      topLevel.length = 0;
    };
    const loadVisuals = async () => {
      clearLoadedVisuals();
      const payload = await getSceneVisual(sess);
      for (const body of payload.bodies) {
        const group = new Object3D();
        group.name = body.name;
        bodyStateRef.current[body.name] = {
          movable: body.movable,
          selectionBody: body.selectionBody,
          matrix: [...body.matrix],
        };
        group.userData.zapdosBody = body.name;
        group.userData.zapdosEditable = body.editable;
        group.userData.zapdosMovable = body.movable;
        group.userData.zapdosSelectionBody = body.selectionBody;
        applyObjectMatrix(group, body.matrix);
        bodyObjectsRef.current[body.name] = group;
        topLevel.push(group);
        root.add(group);
      }
      const loadedMeshes = await loadSceneMeshResources(payload.meshes);
      for (const { geometry, image, item } of loadedMeshes) {
        const mesh = new Mesh(geometry, getSceneMaterial(item, image));
        mesh.castShadow = !item.name.endsWith(".plane");
        mesh.receiveShadow = true;
        mesh.name = item.name;
        if (item.body) {
          const owner = bodyObjectsRef.current[item.body];
          mesh.userData.zapdosBody = item.body;
          mesh.userData.zapdosEditable = owner?.userData.zapdosEditable === true;
          mesh.userData.zapdosMovable = owner?.userData.zapdosMovable === true;
          mesh.userData.zapdosSelectionBody = typeof owner?.userData.zapdosSelectionBody === "string"
            ? owner.userData.zapdosSelectionBody as string
            : null;
          applyObjectMatrix(mesh, item.localMatrix as number[]);
          owner?.add(mesh);
        } else if (item.matrix) {
          applyObjectMatrix(mesh, item.matrix);
          topLevel.push(mesh);
          root.add(mesh);
        }
      }
      const nextBodies = new Set(Object.keys(bodyObjectsRef.current));
      setSelectedBody(current => clearMissingSelection(current, nextBodies));
    };
    sse.onmessage = event => {
      const payload = JSON.parse(event.data) as {
        inactive?: boolean;
        pose?: Record<string, number[]>;
        scene_revision?: string;
      };
      if (isZapdosInactivePayload(payload)) return fail(ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE);
      const nextRevision = getZapdosSceneRevision(payload);
      if (shouldReloadSceneRevision(sceneRevisionRef.current, nextRevision)) {
        sceneRevisionRef.current = nextRevision;
        void loadVisuals().catch(fail);
        return;
      }
      if (!payload.pose) return;
      for (const [name, matrix] of Object.entries(payload.pose)) {
        if (!shouldApplyBodyPose(name, draggingBodyRef.current, bodyStateRef.current)) continue;
        const object = bodyObjectsRef.current[name];
        if (object) {
          applyObjectMatrix(object, matrix);
        }
        const state = bodyStateRef.current[name];
        if (state) {
          state.matrix = [...matrix];
        }
      }
      counter.record();
      if (counter.frame > 100 || Date.now() - counter.start > 1000) setSse(counter.flush());
    };
    sse.onerror = () => fail(ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE);
    void loadVisuals().catch(fail);
    return () => {
      disposed = true;
      sse.close();
      clearInterval(ping);
      clearLoadedVisuals();
    };
  }, [onRuntimeError, scene, sess, setSse]);

  const commitSelection = () => {
    const body = selectedBodyRef.current;
    if (!body || !selectedMovableObject) return;
    const pose = buildBodyPosePayload(selectedMovableObject);
    void setSceneBodyPose(sess, body, pose)
      .catch(error => onRuntimeError(getZapdosRuntimeErrorMessage(error)))
      .finally(() => {
        dragPreviewBodiesRef.current = null;
        draggingBodyRef.current = null;
        setTransformDragging(false);
      });
  };

  return selectedMovableObject ? <TransformControls
    mode={ mode }
    object={ selectedMovableObject }
    onMouseDown={ () => {
      draggingBodyRef.current = selectedBodyRef.current;
      dragPreviewBodiesRef.current = Object.fromEntries(
        Object.entries(bodyStateRef.current)
          .filter(([, state]) => state.selectionBody === selectedBodyRef.current)
          .map(([body, state]) => [body, { ...state, matrix: [...state.matrix] }]),
      );
      setTransformDragging(true);
    } }
    onMouseUp={ commitSelection }
    onObjectChange={ () => {
      selectedMovableObject.updateMatrixWorld(true);
      const previewMatrices = getDraggedBodyMatrices(
        draggingBodyRef.current,
        selectedMovableObject.matrixWorld.toArray(),
        dragPreviewBodiesRef.current ?? {},
      );
      for (const [body, matrix] of Object.entries(previewMatrices)) {
        const object = bodyObjectsRef.current[body];
        if (object) {
          applyObjectMatrix(object, matrix);
        }
      }
    } } /> : null;
}

export function ZapdosScene({
  activeRobotModelKey,
  onRobotModelChange,
  sess,
  onRuntimeError,
}: {
  activeRobotModelKey: RobotModelKey | null
  onRobotModelChange: (key: RobotModelKey) => void
  sess: string
  onRuntimeError: (message: string) => void
}) {
  const [mode, setMode] = useState<ZapdosTransformMode>("translate");
  const [selectedBody, setSelectedBody] = useState<string | null>(null);
  const [sse, setSse] = useState(0);
  const [spark, setSpark] = useState<SparkRenderer | null>(null);
  const [splatRoot, setSplatRoot] = useState<Object3D | null>(null);
  const [splatReady, setSplatReady] = useState(false);
  const [transformDragging, setTransformDragging] = useState(false);
  useZapdosAgentTools(sess);
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
      <SparkRendererBridge ref={ setSpark } />
      { /* TODO: enable splatRoot after resolving issue with environment rendering */ }
      <SceneEnvironment spark={ spark } splatReady={ splatReady } splatRoot={ null } />
      <ambientLight intensity={ 1.2 } />
      <SurfacePivotControls enabled={ !transformDragging } pickRootName={ PIVOT_PICK_ROOT } />
      <EffectComposer multisampling={ 8 }><N8AO aoRadius={ 1 } distanceFalloff={ 1 } intensity={ 4 } /></EffectComposer>
      <group name={ PIVOT_PICK_ROOT }>
        <group position={ [0, 2, 1.2] } rotation={ [-Math.PI*0.55, 0, 0] } ref={ setSplatRoot }>
          {
            /**
             * disabled
            <SparkSplat onReadyChange={ setSplatReady } url="/tmp/point_cloud.ply" />
             */
          }
        </group>
      </group>
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
    <ZapdosCameraStrip onRuntimeError={ onRuntimeError } sess={ sess } />
    <ZapdosTopOverlay
      activeRobotModelKey={ activeRobotModelKey }
      mode={ mode }
      onRobotModelChange={ onRobotModelChange }
      selectedBody={ selectedBody }
      sess={ sess }
      sse={ sse } />
    </Panel>
    <Panel maxSize="33%">
      <CopilotChat />
    </Panel>
  </Group>
}
