"use client"

import { useEffect, useRef, useState } from "react"
import type { Manifold, ManifoldToplevel } from "manifold-3d"
import { buildAntenna, type AntennaOptions } from "./antenna-builder"

interface AntennaProps {
    options?: AntennaOptions
}

function manifoldToGeometry(THREE: typeof import("three"), manifold: Manifold) {
    const mesh = manifold.getMesh()
    const positions = new Float32Array((mesh.vertProperties.length / mesh.numProp) * 3)

    for (let source = 0, target = 0; source < mesh.vertProperties.length; source += mesh.numProp) {
        positions[target] = mesh.vertProperties[source]
        positions[target + 1] = mesh.vertProperties[source + 1]
        positions[target + 2] = mesh.vertProperties[source + 2]
        target += 3
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3))
    geometry.setIndex(new THREE.BufferAttribute(mesh.triVerts, 1))
    geometry.computeVertexNormals()
    geometry.computeBoundingSphere()
    return geometry
}

function meshToManifold(module: ManifoldToplevel, sourceMesh: import("three").Mesh) {
    const geometry = sourceMesh.geometry
    const position =
        geometry.getAttribute("position") as import("three").BufferAttribute | import("three").InterleavedBufferAttribute
    const positionArray = "data" in position ? position.data.array : position.array
    const positionStride = "data" in position ? position.data.stride : position.itemSize
    const positionOffset = "data" in position ? position.offset : 0
    const vertProperties = new Float32Array(position.count * 3)
    const matrix = sourceMesh.matrixWorld.elements

    for (let vertex = 0, target = 0; vertex < position.count; vertex += 1, target += 3) {
        const source = positionOffset + vertex * positionStride
        const x = positionArray[source]
        const y = positionArray[source + 1]
        const z = positionArray[source + 2]
        vertProperties[target] = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]
        vertProperties[target + 1] = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]
        vertProperties[target + 2] = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]
    }

    const index = geometry.getIndex()
    const triVerts = index ? Uint32Array.from(index.array as ArrayLike<number>) : new Uint32Array(position.count)

    if (!index) {
        for (let vertex = 0; vertex < triVerts.length; vertex += 1) {
            triVerts[vertex] = vertex
        }
    }

    if (sourceMesh.matrixWorld.determinant() < 0) {
        for (let tri = 0; tri < triVerts.length; tri += 3) {
            const swap = triVerts[tri + 1]
            triVerts[tri + 1] = triVerts[tri + 2]
            triVerts[tri + 2] = swap
        }
    }

    const mesh = new module.Mesh({
        numProp: 3,
        vertProperties,
        triVerts,
    })
    mesh.merge()
    return new module.Manifold(mesh)
}

export default function Antenna({ options }: AntennaProps = {}) {
    const hostRef = useRef<HTMLDivElement | null>(null)
    const [status, setStatus] = useState<"loading" | "ready" | "error">("loading")
    const sceneRef = useRef<{
        antennaFrame: import("three").Mesh | null
        innerBody: import("three").Mesh | null
        phoneShell: import("three").Group | null
        antennaEnvelope: Manifold | null
        module: ManifoldToplevel | null
        THREE: typeof import("three") | null
    }>({ antennaFrame: null, innerBody: null, phoneShell: null, antennaEnvelope: null, module: null, THREE: null })

    useEffect(() => {
        const host = hostRef.current
        if (!host) {
            return
        }

        let cancelled = false
        let frameId = 0
        let renderer: import("three").WebGLRenderer | null = null
        let resizeObserver: ResizeObserver | null = null
        let controls: import("three/examples/jsm/controls/OrbitControls.js").OrbitControls | null = null

        async function setup() {
            const [
                THREE,
                { OBJLoader },
                { OrbitControls },
                { getManifoldModule },
            ] = await Promise.all([
                import("three"),
                import("three/examples/jsm/loaders/OBJLoader.js"),
                import("three/examples/jsm/controls/OrbitControls.js"),
                import("manifold-3d/lib/wasm.js"),
            ])

            const module = await getManifoldModule()
            const scene = new THREE.Scene()
            scene.background = new THREE.Color("#07111c")
            scene.fog = new THREE.Fog("#07111c", 16, 32)

            const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100)
            camera.position.set(7.8, 5.4, 12.6)

            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
            renderer.shadowMap.enabled = true
            renderer.shadowMap.type = THREE.PCFSoftShadowMap
            host?.appendChild(renderer.domElement)

            const ambientLight = new THREE.HemisphereLight("#cbe7ff", "#08131f", 1.4)
            const keyLight = new THREE.DirectionalLight("#ffffff", 2.2)
            keyLight.position.set(5, 9, 12)
            keyLight.castShadow = true
            keyLight.shadow.mapSize.set(1024, 1024)

            const rimLight = new THREE.DirectionalLight("#7dc1ff", 1.5)
            rimLight.position.set(-8, -3, -7)

            const ground = new THREE.Mesh(
                new THREE.CircleGeometry(16, 96),
                new THREE.MeshBasicMaterial({
                    color: "#0f1f2f",
                    transparent: true,
                    opacity: 0.52,
                })
            )
            ground.rotation.x = -Math.PI / 2
            ground.position.y = -5.7

            scene.add(ambientLight, keyLight, rimLight, ground)

            controls = new OrbitControls(camera, renderer.domElement)
            controls.enablePan = false
            controls.enableDamping = true
            controls.autoRotate = true
            controls.autoRotateSpeed = 0.8
            controls.minDistance = 9
            controls.maxDistance = 18
            controls.target.set(0, 0, 0)

            const shellMaterial = new THREE.MeshPhysicalMaterial({
                color: "#8fa4b8",
                roughness: 0.24,
                metalness: 0.08,
                transparent: true,
                opacity: 0.18,
                transmission: 0.14,
                thickness: 0.4,
                side: THREE.DoubleSide,
            })
            const shellEdgesMaterial = new THREE.LineBasicMaterial({
                color: "#d9e7f2",
                transparent: true,
                opacity: 0.38,
            })

            let phoneShell: import("three").Group
            try {
                const loader = new OBJLoader()
                phoneShell = await loader.loadAsync("/basic-phone.obj")
            } catch {
                phoneShell = new THREE.Group()
                phoneShell.add(new THREE.Mesh(new THREE.BoxGeometry(80, 160, 8)))
            }

            if (cancelled) {
                return
            }

            const rawBounds = new THREE.Box3().setFromObject(phoneShell)
            const rawSize = rawBounds.getSize(new THREE.Vector3())
            const rawCenter = rawBounds.getCenter(new THREE.Vector3())
            const scale = 12 / rawSize.y
            const scaledSize = rawSize.clone().multiplyScalar(scale)

            phoneShell.scale.setScalar(scale)
            phoneShell.position.set(
                -rawCenter.x * scale,
                -rawCenter.y * scale,
                -rawCenter.z * scale,
            )

            const shellMeshes: import("three").Mesh[] = []
            phoneShell.traverse((object) => {
                if (object instanceof THREE.Mesh) {
                    shellMeshes.push(object)
                }
            })
            shellMeshes.forEach((mesh) => {
                mesh.material = shellMaterial
                mesh.castShadow = true
                mesh.receiveShadow = true
                const edges = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry), shellEdgesMaterial)
                mesh.add(edges)
            })

            phoneShell.updateMatrixWorld(true)

            let antennaEnvelope: Manifold | null = null
            for (const shellMesh of shellMeshes) {
                const shell = meshToManifold(module, shellMesh)
                if (!antennaEnvelope) {
                    antennaEnvelope = shell
                    continue
                }

                const combined = antennaEnvelope.add(shell)
                antennaEnvelope.delete()
                shell.delete()
                antennaEnvelope = combined
            }

            if (!antennaEnvelope) {
                throw new Error("Phone shell did not contain any meshes")
            }

            const frameWidth = scaledSize.x * 0.055,
                cutWidth = 0.2,
                nibWidth = 0.3

            scene.add(phoneShell)

            // Store references for dynamic updates
            sceneRef.current.antennaEnvelope = antennaEnvelope
            sceneRef.current.module = module
            sceneRef.current.THREE = THREE
            sceneRef.current.phoneShell = phoneShell

            // Generate initial antenna
            const defaultOptions: AntennaOptions = {
                frame: {
                    width: frameWidth,
                    gap: scaledSize.x * 0.1,
                    cuts: [
                        { position: "top", distance: -scaledSize.x * 0.19, width: cutWidth },
                        { position: "top", distance: scaledSize.x * 0.19, width: cutWidth },
                        { position: "bottom", distance: -scaledSize.x * 0.19, width: cutWidth },
                        { position: "bottom", distance: scaledSize.x * 0.19, width: cutWidth },
                        { position: "left", distance: scaledSize.y * 0.12, width: cutWidth },
                        { position: "right", distance: -scaledSize.y * 0.12, width: cutWidth },
                    ],
                    nibs: [
                        { position: "top", distance: 0, thickness: scaledSize.z, translate: [0, 0, 0], width: nibWidth },
                        { position: "bottom", distance: 0, thickness: scaledSize.z, translate: [0, 0, 0], width: nibWidth },
                        { position: "left", distance: scaledSize.y * 0.18, thickness: scaledSize.z, translate: [0, 0, 0], width: nibWidth },
                        { position: "right", distance: -scaledSize.y * 0.18, thickness: scaledSize.z, translate: [0, 0, 0], width: nibWidth },
                    ],
                },
            }
            const initialOptions = options || defaultOptions
            const antenna = buildAntenna(antennaEnvelope, module, initialOptions)

            const antennaFrameMesh = new THREE.Mesh(
                manifoldToGeometry(THREE, antenna.antennaFrame),
                new THREE.MeshStandardMaterial({
                    color: "#ff9b45",
                    metalness: 0.35,
                    roughness: 0.26,
                    emissive: "#7a2e00",
                    emissiveIntensity: 0.35,
                })
            )
            antennaFrameMesh.castShadow = true
            antennaFrameMesh.receiveShadow = true

            const innerBodyMesh = new THREE.Mesh(
                manifoldToGeometry(THREE, antenna.inner),
                new THREE.MeshStandardMaterial({
                    color: "#49d3d6",
                    metalness: 0.08,
                    roughness: 0.32,
                    emissive: "#113c40",
                    emissiveIntensity: 0.35,
                })
            )
            innerBodyMesh.castShadow = true
            innerBodyMesh.receiveShadow = true

            scene.add(innerBodyMesh, antennaFrameMesh)

            sceneRef.current.antennaFrame = antennaFrameMesh
            sceneRef.current.innerBody = innerBodyMesh

            antenna.antennaFrame.delete()
            antenna.inner.delete()

            const resize = () => {
                if (!renderer) {
                    return
                }

                const width = host?.clientWidth || 1
                const height = host?.clientHeight || 1
                renderer.setSize(width, height, false)
                camera.aspect = width / height
                camera.updateProjectionMatrix()
            }

            resize()
            resizeObserver = new ResizeObserver(resize)
            host && resizeObserver.observe(host)

            const animate = () => {
                if (cancelled) {
                    return
                }

                controls?.update()
                renderer?.render(scene, camera)
                frameId = window.requestAnimationFrame(animate)
            }

            setStatus("ready")
            animate()
        }

        setup().catch((error) => {
            console.error(error)
            setStatus("error")
        })

        return () => {
            cancelled = true
            window.cancelAnimationFrame(frameId)
            resizeObserver?.disconnect()
            controls?.dispose()

            if (renderer) {
                renderer.dispose()
                renderer.forceContextLoss()
                renderer.domElement.remove()
            }
        }
    }, [])

    // Update geometry when options change
    useEffect(() => {
        const { antennaEnvelope, module, THREE } = sceneRef.current
        if (!antennaEnvelope || !module || !THREE || !options) return

        // Remove old meshes
        if (sceneRef.current.antennaFrame) {
            sceneRef.current.antennaFrame.removeFromParent()
            sceneRef.current.antennaFrame.geometry.dispose()
            if (Array.isArray(sceneRef.current.antennaFrame.material)) {
                sceneRef.current.antennaFrame.material.forEach(m => m.dispose())
            } else {
                sceneRef.current.antennaFrame.material.dispose()
            }
        }
        if (sceneRef.current.innerBody) {
            sceneRef.current.innerBody.removeFromParent()
            sceneRef.current.innerBody.geometry.dispose()
            if (Array.isArray(sceneRef.current.innerBody.material)) {
                sceneRef.current.innerBody.material.forEach(m => m.dispose())
            } else {
                sceneRef.current.innerBody.material.dispose()
            }
        }

        // Build new antenna
        const antenna = buildAntenna(antennaEnvelope, module, options)

        const antennaFrame = new THREE.Mesh(
            manifoldToGeometry(THREE, antenna.antennaFrame),
            new THREE.MeshStandardMaterial({
                color: "#ff9b45",
                metalness: 0.35,
                roughness: 0.26,
                emissive: "#7a2e00",
                emissiveIntensity: 0.35,
            })
        )
        antennaFrame.castShadow = true
        antennaFrame.receiveShadow = true

        const innerBody = new THREE.Mesh(
            manifoldToGeometry(THREE, antenna.inner),
            new THREE.MeshStandardMaterial({
                color: "#49d3d6",
                metalness: 0.08,
                roughness: 0.32,
                emissive: "#113c40",
                emissiveIntensity: 0.35,
            })
        )
        innerBody.castShadow = true
        innerBody.receiveShadow = true

        // Add to scene
        sceneRef.current.phoneShell?.parent?.add(antennaFrame)
        sceneRef.current.phoneShell?.parent?.add(innerBody)

        sceneRef.current.antennaFrame = antennaFrame
        sceneRef.current.innerBody = innerBody

        // Cleanup manifolds
        antenna.antennaFrame.delete()
        antenna.inner.delete()
    }, [options])

    return (
        <div
            style={{
                position: "relative",
                width: "100%",
                height: "100%",
                overflow: "hidden",
                background: "radial-gradient(circle at 20% 20%, #13283a, #07111c 58%, #040a12)",
            }}
        >
            <div ref={hostRef} style={{ width: "100%", height: "100%" }} />
            {status !== "ready" ? (
                <div
                    style={{
                        position: "absolute",
                        left: 24,
                        top: 24,
                        padding: "10px 14px",
                        borderRadius: 999,
                        background: "rgba(6, 14, 22, 0.72)",
                        border: "1px solid rgba(138, 182, 212, 0.18)",
                        color: "#d7e8f4",
                        fontSize: 13,
                        letterSpacing: "0.04em",
                        textTransform: "uppercase",
                    }}
                >
                    {status === "loading" ? "Loading antenna model" : "Unable to render antenna model"}
                </div>
            ) : null}
        </div>
    )
}
