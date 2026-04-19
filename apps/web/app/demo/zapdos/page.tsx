'use client'
import { useEffect, useRef } from "react"
import { Canvas, type ThreeEvent, useThree } from "@react-three/fiber"

import { useLocalUUID } from "../../../utils/hooks"
import { SparkRendererBridge, SparkSplat } from "../../../utils/three/splat"
import { OrbitPointTrackballControls, TrackballOrbitControls } from "../../../utils/three/control"

async function call(sess: string, ...args: any[]) {
    const res = await fetch(`/python/zapdos/${sess}/call`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(args)
    })
    return await res.json()
}

function MessageReceiver() {
    const sess = useLocalUUID("zapdos-session"),
        { scene } = useThree() || { }
    useEffect(() => {
        const sse = new EventSource(`/python/zapdos/${sess}/start`),
            ping = setInterval(() => call(sess, "ping"), 30000)
        sse.onmessage = event => {
            if (event.data.pose) {
                // TODO: update the robot pose in the THREE.js scene based on the received data
                scene
            }
        }
        return () => {
            sse.close()
            clearInterval(ping)
        }
    }, [sess])
    return null
}

export default function Zapdos() {
    const controlsRef = useRef<OrbitPointTrackballControls | null>(null)
    const handlePointDown = (event: ThreeEvent<MouseEvent>) => {
        controlsRef.current?.setOrbitPoint(event.point.x, event.point.y, event.point.z)
    }

    return <Canvas className="w-full h-full">
        { /* required to render spark splts */ }
        <SparkRendererBridge />
        <TrackballOrbitControls controlsRef={ controlsRef } />
        <SparkSplat url="/tmp/butterfly.spz" />
        <mesh onPointerDown={ handlePointDown } scale={ [0.1, 0.1, 0.1] } >
            <boxGeometry/>
            <meshNormalMaterial />
        </mesh>
        <MessageReceiver />
    </Canvas>
}
