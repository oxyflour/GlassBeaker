'use client'

import { useEffect, useRef } from "react"
import { useThree } from "@react-three/fiber"
import { PerspectiveCamera, Vector2, Vector3 } from "three"

import {
    createRigState,
    dollyRig,
    finishGesture,
    pickSurfacePoint,
    startGesture,
    updateGesture,
    type SurfacePivotGesture,
    type SurfacePivotRig,
} from "../../utils/surfacePivotMath"

const DEFAULT_PIVOT = new Vector3(0, 0, 0)
const DOLLY_SPEED = 0.002
const MIN_DISTANCE = 0.25
const MAX_DISTANCE = 100

export function SurfacePivotControls() {
    const { camera, gl, invalidate, scene } = useThree()
    const rigRef = useRef<SurfacePivotRig | null>(null)
    const gestureRef = useRef<SurfacePivotGesture | null>(null)
    const pointerIdRef = useRef<number | null>(null)

    useEffect(() => {
        if (!(camera instanceof PerspectiveCamera)) {
            rigRef.current = null
            gestureRef.current = null
            return
        }

        const element = gl.domElement

        const readRig = () => {
            const rig = createRigState(camera.position, camera.quaternion, rigRef.current?.pivot ?? DEFAULT_PIVOT)
            rigRef.current = rig
            return rig
        }

        const applyRig = (rig: SurfacePivotRig) => {
            rigRef.current = rig
            camera.position.copy(rig.position)
            camera.quaternion.copy(rig.quaternion)
            camera.updateMatrixWorld(true)
            invalidate()
        }

        const getPointer = (clientX: number, clientY: number) => {
            const rect = element.getBoundingClientRect()
            return new Vector2(clientX - rect.left, clientY - rect.top)
        }

        const toNdc = (point: Vector2) => new Vector2(
            point.x / element.clientWidth * 2 - 1,
            -(point.y / element.clientHeight) * 2 + 1,
        )

        const onPointerDown = (event: PointerEvent) => {
            if (event.button !== 0 && event.button !== 1) {
                return
            }

            event.preventDefault()
            const point = getPointer(event.clientX, event.clientY)
            let pendingPivot: Vector3 | null = null
            if (event.button === 0) {
                scene.updateMatrixWorld(true)
                camera.updateMatrixWorld(true)
                pendingPivot = pickSurfacePoint(scene, camera, toNdc(point))?.point ?? null
            }

            readRig()
            gestureRef.current = startGesture(event.button, point, pendingPivot)
            pointerIdRef.current = event.pointerId
            element.setPointerCapture(event.pointerId)
        }

        const onPointerMove = (event: PointerEvent) => {
            const gesture = gestureRef.current
            const rig = rigRef.current
            if (!gesture || !rig || event.pointerId !== pointerIdRef.current) {
                return
            }

            const point = getPointer(event.clientX, event.clientY)
            const next = updateGesture(
                rig,
                gesture,
                point,
                { width: element.clientWidth, height: element.clientHeight },
                camera.fov,
                {},
            )
            gestureRef.current = next.gesture
            rigRef.current = next.rig
            if (next.changed) {
                applyRig(next.rig)
            }
        }

        const endGesture = (event: PointerEvent) => {
            const gesture = gestureRef.current
            const rig = rigRef.current
            if (!gesture || !rig || event.pointerId !== pointerIdRef.current) {
                return
            }

            const next = finishGesture(rig, gesture)
            gestureRef.current = null
            pointerIdRef.current = null
            if (element.hasPointerCapture(event.pointerId)) {
                element.releasePointerCapture(event.pointerId)
            }
            applyRig(next)
        }

        const onWheel = (event: WheelEvent) => {
            const rig = rigRef.current ?? readRig()
            event.preventDefault()
            applyRig(dollyRig(rig, event.deltaY, DOLLY_SPEED, MIN_DISTANCE, MAX_DISTANCE))
        }

        readRig()
        element.addEventListener("pointerdown", onPointerDown)
        element.addEventListener("pointermove", onPointerMove)
        element.addEventListener("pointerup", endGesture)
        element.addEventListener("pointercancel", endGesture)
        element.addEventListener("wheel", onWheel, { passive: false })
        return () => {
            element.removeEventListener("pointerdown", onPointerDown)
            element.removeEventListener("pointermove", onPointerMove)
            element.removeEventListener("pointerup", endGesture)
            element.removeEventListener("pointercancel", endGesture)
            element.removeEventListener("wheel", onWheel)
        }
    }, [camera, gl, invalidate, scene])

    return null
}
