'use client'
import { useEffect, useRef } from "react"
import { Canvas, type ThreeEvent, useThree } from "@react-three/fiber"
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js"
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js"
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

async function call<T>(sess: string, method: string, ...args: unknown[]) {
    const res = await fetch(`/python/zapdos/${sess}/call/${method}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(args)
    })
    if (!res.ok) {
        throw new Error(await res.text())
    }
    return await res.json() as T
}

const stlLoader = new STLLoader(),
    objLoader = new OBJLoader()
async function getGeometry(item: RobotVisual) {
    const ext = item.name.split('.').pop()?.toLowerCase() || '',
        { size = [1, 1, 1] } = item
    if (ext === 'stl') {
        return await stlLoader.loadAsync(item.url || '')
    } else if (ext === 'obj') {
        const obj = await objLoader.loadAsync(item.url || '')
        let geometry: BufferGeometry | null = null
        obj.traverse(child => {
            if (child instanceof Mesh && child.geometry) {
                geometry = child.geometry as BufferGeometry
            }
        })
        if (!geometry) {
            throw new Error(`No mesh found in OBJ ${item.url}`)
        }
        return geometry
    } else if (ext === 'box') {
        return new BoxGeometry(size[0], size[1], size[2])
    } else if (ext === 'capsule') {
        const geometry = new CapsuleGeometry(size[0], size[1], 12, 24)
        geometry.rotateX(Math.PI / 2)
        return geometry
    } else if (ext === 'cylinder') {
        const geometry = new CylinderGeometry(size[0], size[0], size[1], 24)
        geometry.rotateX(Math.PI / 2)
        return geometry
    } else if (ext === 'ellipsoid') {
        const geometry = new SphereGeometry(1, 24, 16)
        geometry.scale(size[0] / 2, size[1] / 2, size[2] / 2)
        return geometry
    } else if (ext === 'plane') {
        return new PlaneGeometry(size[0], size[1])
    } else if (ext === 'sphere') {
        return new SphereGeometry(size[0], 24, 16)
    } else {
        throw new Error(`Unsupported geometry type for ${item.name}`)
    }
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
        const sse = new EventSource(`/python/zapdos/${sess}/call/start`),
            ping = setInterval(() => void call(sess, "ping").catch(() => null), 30000),
            added: Record<string, Object3D> = { },
            state = { disposed: false }

        sse.onmessage = event => {
            const { pose = { }, topic, msg } = JSON.parse(event.data) as {
                pose?: RobotPose
                topic?: string
                msg?: any
            }
            for (const [name, matrix] of Object.entries(pose)) {
                const object = added[name]
                if (object) {
                    object.matrix.fromArray(matrix)
                    object.matrixWorldNeedsUpdate = true
                }
            }
            if (topic) {
                console.log('got topic', topic, msg)
            }
        }

        async function loadVisuals() {
            const list = await call<RobotVisual[]>(sess, "get_visual"),
                visuals = await Promise.all(list.map(async item => ({ ...item, geometry: await getGeometry(item) })))
            if (state.disposed) {
                return
            }

            for (const item of visuals) {
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

            await Promise.all([
                call(sess, 'subscribe', '/left_arm', 'sensor_msgs/msg/JointState'),
                call(sess, 'subscribe', '/right_arm', 'sensor_msgs/msg/JointState'),
            ])
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
