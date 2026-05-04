'use client'

import { useEffect, useState } from "react"
import { useSearchParams } from "next/navigation"
import { Canvas, useThree } from "@react-three/fiber"
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js"
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js"
import { EffectComposer, N8AO } from "@react-three/postprocessing"
import { Environment, Lightformer } from "@react-three/drei"
import {
    BoxGeometry,
    BufferGeometry,
    CapsuleGeometry,
    Color,
    CylinderGeometry,
    DoubleSide,
    Mesh,
    MeshStandardMaterial,
    Object3D,
    PlaneGeometry,
    SphereGeometry,
    Texture,
    TextureLoader,
} from "three"

import { useLocalUUID } from "../../../utils/hooks"
import { SparkRendererBridge, SparkSplat } from "../../../utils/three/splat"
import { SurfacePivotControls } from "../../../components/zapdos/SurfacePivotControls"
import { ZapdosTopOverlay } from "../../../components/zapdos/ZapdosTopOverlay"
import {
    buildZapdosInitStreamUrl,
    buildZapdosSessionStorageKey,
    parseZapdosInitEvent,
    type ZapdosInitPhase,
} from "../../../components/zapdos/zapdos-import"
import {
    getZapdosRuntimeErrorMessage,
    isZapdosInactivePayload,
    ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE,
} from "../../../components/zapdos/zapdos-runtime"

const PIVOT_PICK_ROOT = "surface-pivot-content"

interface RobotVisual {
    name: string
    kind: string
    color: [number, number, number, number]
    matrix: number[]
    size?: number[]
    mesh?: string
    texture?: string
}

async function call<T>(sess: string, method: string, ...args: unknown[]) {
    const res = await fetch(`/python/zapdos/${sess}/call/${method}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(args),
    })
    if (!res.ok) {
        throw new Error(await res.text())
    }
    return await res.json() as T
}

const stlLoader = new STLLoader()
const objLoader = new OBJLoader()
const textureLoader = new TextureLoader()
const materials: Record<string, MeshStandardMaterial> = {}

async function getGeometry(item: RobotVisual) {
    const { size = [1, 1, 1] } = item
    if (item.kind === "mesh") {
        if (item.mesh?.endsWith(".stl")) return await stlLoader.loadAsync(item.mesh)
        if (item.mesh?.endsWith(".obj")) {
            const obj = await objLoader.loadAsync(item.mesh)
            let geometry: BufferGeometry | null = null
            obj.traverse(child => {
                if (child instanceof Mesh && child.geometry) geometry = child.geometry as BufferGeometry
            })
            if (geometry) return geometry
        }
        throw new Error(`unknown mesh type ${item.mesh}`)
    }
    if (item.kind === "box") return new BoxGeometry(size[0], size[1], size[2])
    if (item.kind === "capsule") {
        const geometry = new CapsuleGeometry(size[0], size[1], 12, 24)
        geometry.rotateX(Math.PI / 2)
        return geometry
    }
    if (item.kind === "cylinder") {
        const geometry = new CylinderGeometry(size[0], size[0], size[1], 24)
        geometry.rotateX(Math.PI / 2)
        return geometry
    }
    if (item.kind === "ellipsoid") {
        const geometry = new SphereGeometry(1, 24, 16)
        geometry.scale(size[0] / 2, size[1] / 2, size[2] / 2)
        return geometry
    }
    if (item.kind === "plane") return new PlaneGeometry(size[0], size[1])
    if (item.kind === "sphere") return new SphereGeometry(size[0], 24, 16)
    throw new Error(`Unsupported geometry type for ${item.name}`)
}

function getMaterial(item: RobotVisual & { image?: Texture }) {
    const [r, g, b, a] = item.color
    const isPlane = item.name.endsWith(".plane")
    const key = `${item.color.join(",")}:${isPlane}:${item.texture}`
    return materials[key] || (materials[key] = new MeshStandardMaterial({
        color: new Color(r, g, b),
        opacity: a,
        roughness: isPlane ? 1.0 : 0.2,
        metalness: isPlane ? 0.0 : 0.3,
        transparent: a < 1,
        ...(item.image ? { map: item.image } : {}),
        ...(isPlane ? { side: DoubleSide } : {}),
    }))
}

class Counter {
    frame = 0
    start = Date.now()
    constructor() {
    }
    flush() {
        const now = Date.now(),
            fps = this.frame / (now - this.start) * 1000
        this.frame = 0
        this.start = now
        return fps
    }
    record() {
        this.frame += 1
    }
}

function Models({
    sess,
    setStats,
    onRuntimeError,
}: {
    sess: string
    setStats: (stat: { sse: number }) => void
    onRuntimeError: (message: string) => void
}) {
    const { scene } = useThree()
    useEffect(() => {
        const sse = new EventSource(`/python/zapdos/${sess}/call/start`)
        const ping = setInterval(() => void call(sess, "ping").catch(() => null), 30000)
        const added: Record<string, Object3D> = {}
        const root = scene.getObjectByName(PIVOT_PICK_ROOT) ?? scene
        const counter = new Counter()
        let disposed = false
        let failed = false

        function fail(error: unknown) {
            if (disposed || failed) return
            failed = true
            onRuntimeError(getZapdosRuntimeErrorMessage(error))
            sse.close()
        }

        sse.onmessage = event => {
            const payload = JSON.parse(event.data) as {
                inactive?: boolean
                pose?: Record<string, number[]>
                topic?: string
                msg?: unknown
            }
            if (isZapdosInactivePayload(payload)) {
                fail(ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE)
                return
            }
            const { pose, topic, msg } = payload
            if (pose) {
                for (const [name, matrix] of Object.entries(pose)) {
                    const object = added[name]
                    if (!object) continue
                    object.matrix.fromArray(matrix)
                    object.matrixWorldNeedsUpdate = true
                }
                counter.record()
                if (counter.frame > 100 || Date.now() - counter.start > 1000) {
                    setStats({ sse: counter.flush() })
                }
            }
            if (topic) {
                console.log("got topic", topic, msg)
            }
        }
        sse.onerror = () => fail(ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE)

        async function loadVisuals() {
            const list = await call<RobotVisual[]>(sess, "get_visual")
            const visuals = await Promise.all(list.map(async item => ({
                ...item,
                geometry: await getGeometry(item),
                image: item.texture ? await textureLoader.loadAsync(item.texture) : undefined,
            })))
            if (disposed) return
            for (const item of visuals) {
                const mesh = new Mesh(item.geometry, getMaterial(item))
                mesh.castShadow = !item.name.endsWith(".plane")
                mesh.receiveShadow = true
                mesh.matrixAutoUpdate = false
                mesh.name = item.name
                mesh.matrix.fromArray(item.matrix)
                mesh.matrixWorldNeedsUpdate = true
                added[item.name] = mesh
                root.add(mesh)
            }
        }

        loadVisuals().catch(fail)
        return () => {
            disposed = true
            sse.close()
            clearInterval(ping)
            for (const item of Object.values(added)) {
                root.remove(item)
            }
        }
    }, [onRuntimeError, scene, sess])
    return null
}

function Cameras({ sess, onRuntimeError }: { sess: string, onRuntimeError: (message: string) => void }) {
    const [cameras, setCameras] = useState([] as string[])
    useEffect(() => {
        call<RobotVisual[]>(sess, "get_camera")
            .then(res => setCameras(Object.keys(res)))
            .catch(error => onRuntimeError(getZapdosRuntimeErrorMessage(error)))
    }, [onRuntimeError, sess])
    return <div className="absolute bottom-0 left-0 w-full">
    {
        cameras.map(camera => <img
            className="inline"
            key={ camera } alt={ camera }
            style={{ width: 320, height: 240, marginLeft: 8, marginBottom: 8 }}
            src={ `/python/zapdos/${sess}/render/${camera}` } >
        </img>)
    }
    </div>
}

export default function ZapdosInit() {
    const searchParams = useSearchParams()
    const sceneUsd = searchParams.get("scene_usd")
    const robotUsd = searchParams.get("robot_usd")
    const sess = useLocalUUID(buildZapdosSessionStorageKey(sceneUsd, robotUsd))
    const [state, setState] = useState<{ phase: ZapdosInitPhase, message: string }>({
        phase: "loading",
        message: "loading",
    })
    useEffect(() => {
        setState({ phase: "loading", message: "loading" })
        const sse = new EventSource(buildZapdosInitStreamUrl(sess, sceneUsd, robotUsd))
        sse.onmessage = event => {
            const next = parseZapdosInitEvent(event.data)
            setState(next)
            if (next.phase !== "loading") {
                sse.close()
            }
        }
        sse.onerror = () => {
            setState({ phase: "error", message: "Session bootstrap failed" })
            sse.close()
        }
        return () => {
            sse.close()
        }
    }, [robotUsd, sceneUsd, sess])
    return state.phase === 'started' ?
        <Zapdos sess={ sess } onRuntimeError={ message => setState({ phase: "error", message }) } /> :
        <div className="w-full h-full text-center">
            { state.message }
        </div>
}

function Zapdos({ sess, onRuntimeError }: { sess: string, onRuntimeError: (message: string) => void }) {
    const [stats, setStats] = useState({ sse: 0 })
    return <div className="relative h-full w-full">
        <Canvas camera={ { position: [2.5, -2.5, 1.8], fov: 45, near: 0.01, far: 100 } } className="h-full w-full">
            <SparkRendererBridge />
            <ambientLight intensity={ 1.2 } />
            <directionalLight intensity={ 1.8 } position={ [6, -4, 8] } />
            <directionalLight intensity={ 0.8 } position={ [-4, 6, 4] } />
            <SurfacePivotControls pickRootName={ PIVOT_PICK_ROOT } />
            <EffectComposer multisampling={ 8 }>
                <N8AO aoRadius={ 1 } distanceFalloff={ 1 } intensity={ 4 } />
            </EffectComposer>
            <Environment resolution={ 256 }>
                <group rotation={ [-Math.PI / 3, 0, 1] }>
                    <Lightformer form="circle" intensity={ 4 } position={ [0, 5, -9] } rotation-x={ Math.PI / 2 } scale={ 2 } />
                    <Lightformer form="circle" intensity={ 2 } position={ [-5, 1, -1] } rotation-y={ Math.PI / 2 } scale={ 2 } />
                    <Lightformer form="circle" intensity={ 2 } position={ [-5, -1, -1] } rotation-y={ Math.PI / 2 } scale={ 2 } />
                    <Lightformer form="circle" intensity={ 2 } position={ [10, 1, 0] } rotation-y={ -Math.PI / 2 } scale={ 8 } />
                </group>
            </Environment>
            <group name={ PIVOT_PICK_ROOT }>
                <group position={ [0, 0, 2] }>
                    <SparkSplat url="/tmp/butterfly.spz" />
                </group>
            </group>
            { sess && <Models sess={ sess } setStats={ setStats } onRuntimeError={ onRuntimeError } /> }
        </Canvas>
        { sess && <Cameras sess={ sess } onRuntimeError={ onRuntimeError } /> }
        <ZapdosTopOverlay sess={ sess } sse={ stats.sse } />
    </div>
}
