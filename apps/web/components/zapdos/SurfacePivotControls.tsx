'use client'

import { useEffect, useLayoutEffect, useRef } from "react"
import { useThree } from "@react-three/fiber"
import { PerspectiveCamera, Scene, Vector2, Vector3 } from "three"

import {
    createRigState,
    dollyRigAnchoredToPivot,
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

export function SurfacePivotControls({ enabled = true, pickRootName }: { enabled?: boolean; pickRootName?: string }) {
    const { camera, gl, invalidate, scene } = useThree()
    const rigRef = useRef<SurfacePivotRig | null>(null)
    const gestureRef = useRef<SurfacePivotGesture | null>(null)
    const pointerIdRef = useRef<number | null>(null)
    const initializedRef = useRef(false)

    useLayoutEffect(() => {
        if (initializedRef.current || !(camera instanceof PerspectiveCamera)) {
            return
        }

        camera.lookAt(DEFAULT_PIVOT)
        camera.updateMatrixWorld(true)
        rigRef.current = createRigState(camera.position, camera.quaternion, DEFAULT_PIVOT)
        initializedRef.current = true
    }, [camera])

    useEffect(() => {
        if (!(camera instanceof PerspectiveCamera)) {
            rigRef.current = null
            gestureRef.current = null
            pointerIdRef.current = null
            return
        }

        const element = gl.domElement

        const clearGesture = () => {
            gestureRef.current = null
            pointerIdRef.current = null
        }

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

        const resolvePendingPivot = (point: Vector2) => {
            const pickRoot = pickRootName ? scene.getObjectByName(pickRootName) : null
            if (pickRootName && !pickRoot) {
                return null
            }

            const hit = pickSurfacePoint((pickRoot ?? scene) as Scene, camera, toNdc(point))
            return hit?.point ?? null
        }

        const isButtonStillPressed = (gesture: SurfacePivotGesture, buttons: number) =>
            gesture.button === 0 ? (buttons & 1) !== 0 : (buttons & 4) !== 0

        const releasePointerCapture = (pointerId: number) => {
            if (element.hasPointerCapture(pointerId)) {
                element.releasePointerCapture(pointerId)
            }
        }

        const onPointerDown = (event: PointerEvent) => {
            if (!enabled) {
                return
            }
            if (event.pointerType !== "mouse" || (event.button !== 0 && event.button !== 1)) {
                return
            }

            event.preventDefault()
            const point = getPointer(event.clientX, event.clientY)
            let pendingPivot: Vector3 | null = null
            if (event.button === 0) {
                scene.updateMatrixWorld(true)
                camera.updateMatrixWorld(true)
                pendingPivot = resolvePendingPivot(point)
            }

            readRig()
            gestureRef.current = startGesture(event.button, point, pendingPivot)
            pointerIdRef.current = event.pointerId
            element.setPointerCapture(event.pointerId)
        }

        const onPointerMove = (event: PointerEvent) => {
            const gesture = gestureRef.current
            const rig = rigRef.current
            if (!gesture || !rig || event.pointerType !== "mouse" || event.pointerId !== pointerIdRef.current || !element.hasPointerCapture(event.pointerId)) {
                return
            }
            if (!isButtonStillPressed(gesture, event.buttons)) {
                const next = finishGesture(rig, gesture)
                releasePointerCapture(event.pointerId)
                clearGesture()
                applyRig(next)
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
            if (!gesture || !rig || event.pointerType !== "mouse" || event.pointerId !== pointerIdRef.current) {
                return
            }

            const next = finishGesture(rig, gesture)
            releasePointerCapture(event.pointerId)
            clearGesture()
            applyRig(next)
        }

        const onLostPointerCapture = (event: PointerEvent) => {
            if (event.pointerType === "mouse" && event.pointerId === pointerIdRef.current) {
                clearGesture()
            }
        }

        const onWheel = (event: WheelEvent) => {
            if (!enabled) {
                return
            }
            const rig = rigRef.current ?? readRig()
            event.preventDefault()
            applyRig(dollyRigAnchoredToPivot(rig, event.deltaY, DOLLY_SPEED, MIN_DISTANCE, MAX_DISTANCE))
        }

        readRig()
        element.addEventListener("pointerdown", onPointerDown)
        element.addEventListener("pointermove", onPointerMove)
        element.addEventListener("pointerup", endGesture)
        element.addEventListener("pointercancel", endGesture)
        element.addEventListener("lostpointercapture", onLostPointerCapture)
        element.addEventListener("wheel", onWheel, { passive: false })
        return () => {
            if (pointerIdRef.current !== null) {
                releasePointerCapture(pointerIdRef.current)
            }
            clearGesture()
            element.removeEventListener("pointerdown", onPointerDown)
            element.removeEventListener("pointermove", onPointerMove)
            element.removeEventListener("pointerup", endGesture)
            element.removeEventListener("pointercancel", endGesture)
            element.removeEventListener("lostpointercapture", onLostPointerCapture)
            element.removeEventListener("wheel", onWheel)
        }
    }, [camera, enabled, gl, invalidate, pickRootName, scene])

    return null
}
