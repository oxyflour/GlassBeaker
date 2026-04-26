'use client'
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Canvas, type ThreeEvent, useFrame, useThree } from "@react-three/fiber"
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js"
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js"
import { EffectComposer, N8AO } from '@react-three/postprocessing'

import { useLocalUUID } from "../../../utils/hooks"
import { SparkRendererBridge, SparkSplat } from "../../../utils/three/splat"
import { BoxGeometry, BufferGeometry, Camera, CapsuleGeometry, Color, CylinderGeometry, DoubleSide, Mesh, MeshStandardMaterial, Object3D, PerspectiveCamera, PlaneGeometry, SphereGeometry, Texture, TextureLoader } from "three"
import { ArcballControls, Environment, Lightformer, OrbitControls } from "@react-three/drei"

type RobotPose = Record<string, number[]>

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
    const { size = [1, 1, 1] } = item
    if (item.kind === 'mesh') {
        if (item.mesh?.endsWith('.stl')) {
            return await stlLoader.loadAsync(item.mesh || '')
        } else if (item.mesh?.endsWith('.obj')) {
            const obj = await objLoader.loadAsync(item.mesh || '')
            let geometry: BufferGeometry | null = null
            obj.traverse(child => {
                if (child instanceof Mesh && child.geometry) {
                    geometry = child.geometry as BufferGeometry
                }
            })
            if (!geometry) {
                throw new Error(`No mesh found in OBJ ${item.mesh}`)
            }
            return geometry
        } else {
            throw Error(`unknown mesh type ${item.mesh}`)
        }
    } else if (item.kind === 'box') {
        return new BoxGeometry(size[0], size[1], size[2])
    } else if (item.kind === 'capsule') {
        const geometry = new CapsuleGeometry(size[0], size[1], 12, 24)
        geometry.rotateX(Math.PI / 2)
        return geometry
    } else if (item.kind === 'cylinder') {
        const geometry = new CylinderGeometry(size[0], size[0], size[1], 24)
        geometry.rotateX(Math.PI / 2)
        return geometry
    } else if (item.kind === 'ellipsoid') {
        const geometry = new SphereGeometry(1, 24, 16)
        geometry.scale(size[0] / 2, size[1] / 2, size[2] / 2)
        return geometry
    } else if (item.kind === 'plane') {
        const geometry = new PlaneGeometry(size[0], size[1])
        return geometry
    } else if (item.kind === 'sphere') {
        return new SphereGeometry(size[0], 24, 16)
    } else {
        throw new Error(`Unsupported geometry type for ${item.name}`)
    }
}

const materials = { } as Record<string, MeshStandardMaterial>,
    textureLoader = new TextureLoader()
function getMaterial(item: RobotVisual & { image?: Texture }) {
    const [r, g, b, a] = item.color,
        isPlane = item.name.endsWith(".plane"),
        key = `${item.color.join(',')}:${isPlane}:${item.texture}`
    return materials[key] || (materials[key] = new MeshStandardMaterial({
        color: new Color(r, g, b),
        opacity: a,
        roughness: isPlane ? 1.0 : 0.2,
        metalness: isPlane ? 0.0 : 0.3,
        transparent: a < 1,
        ...(item.image ? { map: item.image } : { }),
        ...(isPlane ? { side: DoubleSide } : { }),
    }))
}

function ZapdosLoader({ updateCamera }: { updateCamera: (c: Record<string, number[]>) => void }) {
    const sess = useLocalUUID("zapdos-session"),
        { scene } = useThree()

    useEffect(() => {
        const sse = new EventSource(`/python/zapdos/${sess}/call/start`),
            ping = setInterval(() => void call(sess, "ping").catch(() => null), 30000),
            added: Record<string, Object3D> = { },
            state = { disposed: false }

        sse.onmessage = event => {
            const { pose = { }, topic, msg, camera } = JSON.parse(event.data) as {
                pose?: RobotPose
                camera?: Record<string, number[]>
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
            if (camera) {
                updateCamera(camera)
            }
            if (topic) {
                console.log('got topic', topic, msg)
            }
        }

        async function loadVisuals() {
            const list = await call<RobotVisual[]>(sess, "get_visual"),
                visuals = await Promise.all(list.map(async item => ({
                    ...item,
                    geometry: await getGeometry(item),
                    image: item.texture ? await textureLoader.loadAsync(item.texture) : undefined,
                })))
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

interface CameraViewport {
    viewport: { x: number, y: number, width: number, height: number },
    camera: PerspectiveCamera
}

function MultiViewRenderer({ main, cameras }: { main: PerspectiveCamera, cameras: Record<string, CameraViewport> }) {
    const { gl, scene, size } = useThree()
    useFrame(() => {
        const w = size.width
        const h = size.height

        gl.autoClear = false
        gl.clear()
        gl.setScissorTest(true)

        // 主视图：全屏
        gl.setViewport(0, 0, w, h)
        gl.setScissor(0, 0, w, h)
        gl.setScissorTest(true)
        main.updateProjectionMatrix()
        gl.render(scene, main)

        for (const { viewport, camera } of Object.values(cameras)) {
            const { x, y, width, height } = viewport
            gl.clearDepth()
            gl.setViewport(x, y, width, height)
            gl.setScissor(x, y, width, height)
            gl.render(scene, camera)
        }
    }, 1)
    return null
}

const mainCamera = new PerspectiveCamera(45, innerWidth / innerHeight, 0.01, 100)
mainCamera.position.set(2.5, -2.5, 1.8)
export default function Zapdos() {
    const [cameras, setCameras] = useState({ } as Record<string, CameraViewport>),
        // FIXME
        cameraRef = useRef(cameras),
        updateCamera = useCallback((update: Record<string, number[]>) => {
            const cameras = cameraRef.current
            for (const [name, matrix] of Object.entries(update)) {
                const { camera } = cameras[name] || { }
                if (camera) {
                    camera.matrixAutoUpdate = false
                    camera.matrix.fromArray(matrix)
                    camera.matrix.decompose(camera.position, camera.quaternion, camera.scale)
                    camera.updateMatrixWorld(true)
                }
            }
            if (Object.keys(cameraRef.current).sort().join(',') !== Object.keys(update).sort().join(',')) {
                setCameras(Object.fromEntries(Object.keys(update).map((name, idx) => [name, {
                    camera: new PerspectiveCamera(45, 1, 0.01, 100),
                    viewport: { x: 10 + 160 * idx, y: 10, width: 150, height: 150 },
                }])))
            }
        }, [])
    cameraRef.current = cameras

    return <div className="relative w-full h-full">
        <Canvas className="w-full h-full" >
            { /* required to render spark splts */ }
            <SparkRendererBridge />

            <MultiViewRenderer main={ mainCamera } cameras={ cameras } />

            <ambientLight intensity={ 1.2 } />
            <directionalLight intensity={ 1.8 } position={ [6, -4, 8] } />
            <directionalLight intensity={ 0.8 } position={ [-4, 6, 4] } />

            <OrbitControls camera={ mainCamera } />

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

            <group position={ [0, 0, 2] }>
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
            <ZapdosLoader updateCamera={ updateCamera } />
        </Canvas>
    </div>
}
