'use client'
import { useEffect, useRef } from "react"
import { Canvas, type ThreeEvent, useThree } from "@react-three/fiber"
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js"
import { EffectComposer, N8AO } from '@react-three/postprocessing'

import { useLocalUUID } from "../../../utils/hooks"
import { SparkRendererBridge, SparkSplat } from "../../../utils/three/splat"
import { OrbitPointTrackballControls, TrackballOrbitControls } from "../../../utils/three/control"
import { BoxGeometry, BufferGeometry, CapsuleGeometry, Color, CylinderGeometry, DoubleSide, Mesh, MeshStandardMaterial, Object3D, PlaneGeometry, SphereGeometry } from "three"
import { Environment, Lightformer } from "@react-three/drei"

type RobotPose = Record<string, number[]>

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
    stl: new STLLoader(),
} as Record<string, { loadAsync: (url: string) => Promise<BufferGeometry> }>
const primitiveLoaders = {
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
} as Record<string, (size: number[]) => BufferGeometry>
async function getGeometry(item: RobotVisual) {
    const ext = item.name.split('.').pop()?.toLowerCase() || ''
    return item.url ?
        await meshLoaders[ext]?.loadAsync(item.url || '') :
        primitiveLoaders[ext]?.(item.size || [1, 1, 1])
}

const materials = { } as Record<string, MeshStandardMaterial>
function getMaterial(item: RobotVisual) {
    const [r, g, b, a] = item.color,
        isPlane = item.name.endsWith(".plane"),
        key = `${item.color.join(',')}:${isPlane}`
    return materials[key] || (materials[key] = new MeshStandardMaterial({
        color: new Color(r, g, b),
        opacity: a,
        roughness: isPlane ? 1.0 : 0.2,
        metalness: isPlane ? 0.0 : 0.3,
        transparent: a < 1,
        ...(isPlane ? { side: DoubleSide } : { }),
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
                visuals = await Promise.all(list.map(async item => ({ ...item, geometry: await getGeometry(item) })))
            if (!state.disposed) for (const item of visuals) {
                const mesh = new Mesh(item.geometry, getMaterial(item))
                mesh.castShadow = !item.name.endsWith(".plane")
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

        <EffectComposer multisampling={8}>
            <N8AO distanceFalloff={1} aoRadius={1} intensity={4} />
        </EffectComposer>

        <Environment resolution={256}>
            <group rotation={[-Math.PI / 3, 0, 1]}>
                <Lightformer form="circle" intensity={4} rotation-x={Math.PI / 2} position={[0, 5, -9]} scale={2} />
                <Lightformer form="circle" intensity={2} rotation-y={Math.PI / 2} position={[-5, 1, -1]} scale={2} />
                <Lightformer form="circle" intensity={2} rotation-y={Math.PI / 2} position={[-5, -1, -1]} scale={2} />
                <Lightformer form="circle" intensity={2} rotation-y={-Math.PI / 2} position={[10, 1, 0]} scale={8} />
            </group>
        </Environment>

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
