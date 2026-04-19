'use client'
import { useEffect, useRef } from "react"
import { Canvas, type ThreeEvent, useThree } from "@react-three/fiber"
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js"

import { useLocalUUID } from "../../../utils/hooks"
import { SparkRendererBridge, SparkSplat } from "../../../utils/three/splat"
import { OrbitPointTrackballControls, TrackballOrbitControls } from "../../../utils/three/control"
import { BufferGeometry, Color, Mesh, MeshStandardMaterial, Object3D } from "three"

type RobotPose = Record<string, number[]>

interface RobotVisual {
    color: [number, number, number, number]
    matrix: number[]
    name: string
    url: string
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

const loaders = {
    stl: new STLLoader()
} as Record<string, { loadAsync: (url: string) => Promise<BufferGeometry> }>
function getGeometry(url: string) {
    const ext = url.split('.').pop()?.toLowerCase()
    if (!ext || !loaders[ext]) {
        throw new Error(`Unsupported geometry format: ${ext}`)
    }
    return loaders[ext].loadAsync(url)
}

const materials = { } as Record<string, MeshStandardMaterial>
function getMaterial(color: [number, number, number, number]) {
    const [r, g, b, a] = color,
        key = color.join(',')
    return materials[key] || (materials[key] = new MeshStandardMaterial({
        color: new Color(r, g, b),
        opacity: a,
        transparent: a < 1,
    }))
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
                visuals = await Promise.all(list.map(async item => ({ ...item, geometry: await getGeometry(item.url) })))
            if (!state.disposed) for (const { geometry, name, matrix, color } of visuals) {
                const mesh = new Mesh(geometry, getMaterial(color))
                mesh.castShadow = true
                mesh.receiveShadow = true
                mesh.matrixAutoUpdate = false
                mesh.name = name
                mesh.matrix.fromArray(matrix)
                mesh.matrixWorldNeedsUpdate = true
                added[name] = mesh
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
