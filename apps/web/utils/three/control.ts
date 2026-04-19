'use client'
import { RefObject, useEffect, useMemo } from "react"
import { useFrame, useThree } from "@react-three/fiber"
import { Vector3, type OrthographicCamera, type PerspectiveCamera } from "three"
import { TrackballControls as StdlibTrackballControls } from "three-stdlib"

export class OrbitPointTrackballControls extends StdlibTrackballControls {
    focalOffset = new Vector3()
    offsetWorld = new Vector3()
    xColumn = new Vector3()
    yColumn = new Vector3()
    zColumn = new Vector3()
    cameraToPoint = new Vector3()
    localPoint = new Vector3()

    constructor(camera: PerspectiveCamera | OrthographicCamera) {
        super(camera)
        const updateBase = this.update,
            resetBase = this.reset
        this.update = () => {
            if (this.focalOffset.lengthSq()) {
                this.object.position.sub(this.getWorldOffset(this.offsetWorld))
            }
            updateBase()
            if (this.focalOffset.lengthSq()) {
                this.object.position.add(this.getWorldOffset(this.offsetWorld))
                this.object.updateMatrixWorld()
            }
        }
        this.reset = () => {
            this.focalOffset.set(0, 0, 0)
            resetBase()
        }
    }

    getWorldOffset(out: Vector3) {
        this.object.updateMatrixWorld()
        return out.set(0, 0, 0)
            .add(this.xColumn.setFromMatrixColumn(this.object.matrixWorld, 0).multiplyScalar(this.focalOffset.x))
            .add(this.yColumn.setFromMatrixColumn(this.object.matrixWorld, 1).multiplyScalar(-this.focalOffset.y))
            .add(this.zColumn.setFromMatrixColumn(this.object.matrixWorld, 2).multiplyScalar(this.focalOffset.z))
    }

    setOrbitPoint(x: number, y: number, z: number) {
        this.object.updateMatrixWorld()
        const distance = this.cameraToPoint.set(x, y, z).distanceTo(this.object.position)
        this.cameraToPoint.sub(this.object.position)
        this.localPoint.copy(this.xColumn.setFromMatrixColumn(this.object.matrixWorldInverse, 0).multiplyScalar(this.cameraToPoint.x))
        this.localPoint.add(this.yColumn.setFromMatrixColumn(this.object.matrixWorldInverse, 1).multiplyScalar(this.cameraToPoint.y))
        this.localPoint.add(this.zColumn.setFromMatrixColumn(this.object.matrixWorldInverse, 2).multiplyScalar(this.cameraToPoint.z))
        this.localPoint.z += distance
        this.focalOffset.set(-this.localPoint.x, this.localPoint.y, -this.localPoint.z)
        this.target.set(x, y, z)
        this.update()
    }
}

export function TrackballOrbitControls({ controlsRef }: {
    controlsRef: RefObject<OrbitPointTrackballControls | null>
}) {
    const { camera, events, gl, invalidate, viewport } = useThree()
    const controls = useMemo(() => new OrbitPointTrackballControls(camera as PerspectiveCamera | OrthographicCamera), [camera])
    const domElement = events.connected || gl.domElement

    useFrame(() => {
        if (controls.enabled) {
            controls.update()
        }
    }, -1)

    useEffect(() => {
        controls.connect(domElement)
        controlsRef.current = controls
        return () => {
            if (controlsRef.current === controls) {
                controlsRef.current = null
            }
            controls.dispose()
        }
    }, [controls, controlsRef, domElement])

    useEffect(() => {
        const onChange = () => invalidate()
        controls.addEventListener("change", onChange)
        return () => {
            controls.removeEventListener("change", onChange)
        }
    }, [controls, invalidate])

    useEffect(() => {
        controls.handleResize()
    }, [controls, viewport])

    return null
}
