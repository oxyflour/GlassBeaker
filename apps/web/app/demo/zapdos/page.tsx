'use client'
import { useEffect, useRef } from "react"
import { Canvas, type ThreeEvent, useThree } from "@react-three/fiber"
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js"

import { useLocalUUID } from "../../../utils/hooks"
import { SparkRendererBridge, SparkSplat } from "../../../utils/three/splat"
import { OrbitPointTrackballControls, TrackballOrbitControls } from "../../../utils/three/control"
import { BoxGeometry, BufferGeometry, CapsuleGeometry, Color, CylinderGeometry, DoubleSide, Mesh, MeshStandardMaterial, Object3D, PlaneGeometry, SphereGeometry } from "three"

type RobotPose = Record<string, number[]>
type PrimitiveKind = "box" | "capsule" | "cylinder" | "ellipsoid" | "plane" | "sphere"
interface RobotVisual {
    color: [number, number, number, number]
    matrix: number[]
    name: string
    size?: number[]
    url?: string
}

async function call<T>(sess: string, ...args: unknown[]) {
    const res = await fetch(`/python/zapdos/${sess}/call`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(args)
    })
    if (!res.ok) {
        throw new Error(await res.text())
    }
    return await res.json() as T
}

const meshLoaders = {
    stl: new STLLoader()
} as Record<string, { loadAsync: (url: string) => Promise<BufferGeometry> }>
const primitiveLoaders: Record<PrimitiveKind, (size: number[]) => BufferGeometry> = {
    box: size => new BoxGeometry(size[0], size[1], size[2]),
    capsule: size => {
        const geometry = new CapsuleGeometry(size[0], size[1], 12, 24)
        geometry.rotateX(Math.PI / 2)
        return geometry
    },
    cylinder: size => {
        const geometry = new CylinderGeometry(size[0], size[0], size[1], 24)
        geometry.rotateX(Math.PI / 2)
        return geometry
    },
    ellipsoid: size => {
        const geometry = new SphereGeometry(1, 24, 16)
        geometry.scale(size[0] / 2, size[1] / 2, size[2] / 2)
        return geometry
    },
    plane: size => new PlaneGeometry(size[0], size[1]),
    sphere: size => new SphereGeometry(size[0], 24, 16),
}
function getGeometryKind(name: string) {
    const kind = name.split('.').pop()?.toLowerCase()
    if (!kind) {
        throw new Error(`Missing geometry kind in name: ${name}`)
    }
    return kind
}
function getMeshGeometry(kind: string, url: string) {
    if (!meshLoaders[kind]) {
        throw new Error(`Unsupported geometry format: ${kind}`)
    }
    return meshLoaders[kind].loadAsync(url)
}
function isPrimitiveKind(kind: string): kind is PrimitiveKind {
    return kind in primitiveLoaders
}
const primitiveGeometries = { } as Record<string, BufferGeometry>
function getPrimitiveGeometry(kind: PrimitiveKind, size: number[]) {
    const key = `${kind}:${size.join(',')}`
    return primitiveGeometries[key] || (primitiveGeometries[key] = primitiveLoaders[kind](size))
}
async function getGeometry(item: RobotVisual, kind: string) {
    if (item.url) {
        return await getMeshGeometry(kind, item.url)
    }
    if (!isPrimitiveKind(kind) || !item.size) {
        throw new Error(`Unsupported primitive visual: ${item.name}`)
    }
    return getPrimitiveGeometry(kind, item.size)
}

const materials = { } as Record<string, MeshStandardMaterial>
function getMaterial(color: [number, number, number, number], doubleSided = false) {
    const [r, g, b, a] = color,
        key = `${color.join(',')}:${doubleSided ? 'double' : 'front'}`
    const options = {
        color: new Color(r, g, b),
        opacity: a,
        transparent: a < 1,
    }
    return materials[key] || (materials[key] = new MeshStandardMaterial(doubleSided ? { ...options, side: DoubleSide } : options))
}

function ZapdosLoader() {
    const sess = useLocalUUID("zapdos-session"),
        { scene } = useThree()

    useEffect(() => {
        const sse = new EventSource(`/python/zapdos/${sess}/start`),
            ping = setInterval(() => void call(sess, "ping").catch(() => null), 30000),
            added: Record<string, Object3D> = { },
            state = { disposed: false }

        sse.onmessage = event => {
            const { pose = { } } = JSON.parse(event.data) as { pose?: RobotPose }
            for (const [name, matrix] of Object.entries(pose)) {
                const object = added[name]
                if (object) {
                    object.matrix.fromArray(matrix)
                    object.matrixWorldNeedsUpdate = true
                }
            }
        }

        async function loadVisuals() {
            const list = await call<RobotVisual[]>(sess, "get_visual"),
                visuals = await Promise.all(list.map(async item => {
                    const kind = getGeometryKind(item.name)
                    return { ...item, kind, geometry: await getGeometry(item, kind) }
                }))
            if (!state.disposed) for (const item of visuals) {
                const mesh = new Mesh(item.geometry, getMaterial(item.color, item.kind === "plane"))
                mesh.castShadow = item.kind !== "plane"
                mesh.receiveShadow = true
                mesh.matrixAutoUpdate = false
                mesh.name = item.name
                mesh.matrix.fromArray(item.matrix)
                mesh.matrixWorldNeedsUpdate = true
                added[item.name] = mesh
                scene.add(mesh)
            }
        }

        loadVisuals().catch(console.error)
        return () => {
            state.disposed = true
            sse.close()
            clearInterval(ping)
            for (const item of Object.values(added)) {
                scene.remove(item)
            }
        }
    }, [scene, sess])

    return null
}

export default function Zapdos() {
    const controlsRef = useRef<OrbitPointTrackballControls | null>(null)
    const handlePointDown = (event: ThreeEvent<MouseEvent>) => {
        controlsRef.current?.setOrbitPoint(event.point.x, event.point.y, event.point.z)
    }

    return <Canvas
        camera={ { far: 100, near: 0.01, position: [2.5, -2.5, 1.8], up: [0, 0, 1] } }
        className="w-full h-full" >

        { /* required to render spark splts */ }
        <SparkRendererBridge />

        <ambientLight intensity={ 1.2 } />
        <directionalLight intensity={ 1.8 } position={ [6, -4, 8] } />
        <directionalLight intensity={ 0.8 } position={ [-4, 6, 4] } />

        <TrackballOrbitControls controlsRef={ controlsRef } />

        <group position={ [0, 0, 1] }>
            <SparkSplat url="/tmp/butterfly.spz" />
        </group>
        {
            /*
                <mesh onPointerDown={ handlePointDown } scale={ [0.1, 0.1, 0.1] } >
                    <boxGeometry/>
                    <meshNormalMaterial />
                </mesh>
             */
        }
        <ZapdosLoader />
    </Canvas>
}
