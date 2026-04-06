"use client"

import { useEffect, useRef, useState } from "react"

export type JointControl = { kind: string, lower: number, name: string, upper: number }
export type ControlGroup = { joints: JointControl[], name: string }
export type NanamiEvent = { type: string, [key: string]: any }

const CAMERA = { target_xyz: [0, 0, 1.1], yaw_deg: 35, pitch_deg: -18, distance: 3.2 }
const SOURCE = { robot_path: "R1" }

function messageOf(error: unknown, fallback: string) {
    return error instanceof Error ? error.message : fallback
}

function loadStatusText(status?: string) {
    if (status === "queued") return "Waiting for Unreal to start"
    if (status === "loaded") return "Robot loaded"
    return "Loading robot"
}

function emptyValues(groups: ControlGroup[]) {
    return Object.fromEntries(groups.flatMap((group) => group.joints.map((joint) => [joint.name, 0])))
}

export function pretty(name: string) {
    return name.replaceAll("_", " ").replaceAll("joint", "joint ").replace(/\s+/g, " ").trim()
}

export function useSessionFlow(autoRun = false) {
    const [sessionId, setSessionId] = useState("")
    const [previewUrl, setPreviewUrl] = useState("")
    const [controls, setControls] = useState<ControlGroup[]>([])
    const [values, setValues] = useState<Record<string, number>>({})
    const [events, setEvents] = useState<NanamiEvent[]>([])
    const [status, setStatus] = useState("Idle")
    const sourceRef = useRef<EventSource | null>(null)
    const sessionRef = useRef("")
    const autoLoadedRef = useRef(false)
    const autoExportedRef = useRef(false)
    const createRunRef = useRef(0)

    async function post(path: string, body: object) {
        const response = await fetch(`/python/nanami/${path}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        })
        if (!response.ok) {
            throw new Error(await response.text())
        }
        return response.json()
    }

    function resetFlow() {
        autoLoadedRef.current = false
        autoExportedRef.current = false
    }

    function resetSessionState() {
        setSessionId("")
        setPreviewUrl("")
        setControls([])
        setValues({})
    }

    function handleEvent(event: NanamiEvent, activeSessionId: string) {
        if (sessionRef.current !== activeSessionId) return
        setEvents((items) => [event, ...items].slice(0, 10))
        if (event.type === "session_ready") setStatus("Session ready")
        if (event.type === "assets_cached") setStatus((current) => current.startsWith("Waiting") ? current : event.cache_hit ? "Assets cached (hit)" : "Assets cached")
        if (event.type === "robot_load_started") setStatus("Loading robot")
        if (event.type === "robot_loaded") {
            const nextControls = event.controls || []
            setStatus("Robot loaded")
            setControls(nextControls)
            setValues(emptyValues(nextControls))
        }
        if (event.type === "export_started") setStatus(`Exporting ${event.job_id}`)
        if (event.type === "export_done") setStatus(event.result_path ? `Exported: ${event.result_path}` : "Export finished")
        if (event.type === "worker_error") setStatus(event.message || "Worker error")
    }

    async function destroySession(targetId = sessionRef.current) {
        if (!targetId) return
        const isCurrent = sessionRef.current === targetId
        if (isCurrent) {
            sessionRef.current = ""
            sourceRef.current?.close()
            sourceRef.current = null
            resetFlow()
            resetSessionState()
        }
        await post("session/destroy", { session_id: targetId })
        if (isCurrent) setStatus("Session destroyed")
    }

    async function createSession() {
        const createRun = createRunRef.current + 1
        createRunRef.current = createRun
        if (sessionRef.current) {
            await destroySession(sessionRef.current).catch(() => undefined)
        }
        resetFlow()
        resetSessionState()
        setEvents([])
        setStatus("Creating session")
        const data = await post("session/create", { source: SOURCE })
        const activeSessionId = data.session_id as string
        if (createRun !== createRunRef.current) {
            await post("session/destroy", { session_id: activeSessionId }).catch(() => undefined)
            return activeSessionId
        }
        sourceRef.current?.close()
        const source = new EventSource(data.events_url)
        source.onmessage = (message) => handleEvent(JSON.parse(message.data) as NanamiEvent, activeSessionId)
        source.onerror = () => {
            if (sessionRef.current === activeSessionId) setStatus("Events disconnected")
        }
        sourceRef.current = source
        sessionRef.current = activeSessionId
        setSessionId(activeSessionId)
        setPreviewUrl(data.preview_url)
        setStatus("Starting Unreal worker")
        return activeSessionId
    }

    async function loadRobot(targetId = sessionRef.current) {
        if (!targetId) return
        setStatus("Loading robot")
        const data = await post("robot/load", { session_id: targetId }) as { status?: string }
        setStatus(loadStatusText(data.status))
    }

    async function exportFinal(targetId = sessionRef.current) {
        if (!targetId) return
        setStatus("Queueing final export")
        await post("export/start", { session_id: targetId, profile: "final" })
    }

    useEffect(() => {
        if (!autoRun) return
        let active = true
        void createSession()
            .then((createdSessionId) => {
                if (!active) void destroySession(createdSessionId).catch(() => undefined)
            })
            .catch((error) => {
                if (active) setStatus(messageOf(error, "Create session failed"))
            })
        return () => {
            active = false
            createRunRef.current += 1
            if (sessionRef.current) void destroySession(sessionRef.current).catch(() => undefined)
            sourceRef.current?.close()
        }
    }, [autoRun])

    useEffect(() => {
        if (!autoRun || !sessionId || autoLoadedRef.current) return
        autoLoadedRef.current = true
        void loadRobot(sessionId).catch((error) => {
            autoLoadedRef.current = false
            setStatus(messageOf(error, "Load robot failed"))
        })
    }, [autoRun, sessionId])

    useEffect(() => {
        if (!sessionId || controls.length === 0) return
        const timer = window.setTimeout(() => {
            const shouldExport = autoRun && !autoExportedRef.current
            void post("state/update", { session_id: sessionId, joints: values, camera: CAMERA })
                .then(async () => {
                    if (!shouldExport || autoExportedRef.current || sessionRef.current !== sessionId) return
                    autoExportedRef.current = true
                    await exportFinal(sessionId)
                })
                .catch((error) => {
                    if (shouldExport) autoExportedRef.current = false
                    setStatus(messageOf(error, "State update failed"))
                })
        }, 50)
        return () => window.clearTimeout(timer)
    }, [autoRun, controls.length, sessionId, values])

    return {
        controls,
        createSession,
        destroySession,
        events,
        exportFinal,
        loadRobot,
        previewUrl,
        sessionId,
        setValues,
        status,
        values,
    }
}
