"use client"

import { pretty, useSessionFlow } from "./use-session-flow"

type NanamiPanelProps = { autoRun?: boolean }

export default function NanamiPanel({ autoRun = false }: NanamiPanelProps) {
    const {
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
    } = useSessionFlow(autoRun)

    return (
        <div data-testid="nanami-panel" style={{ display: "grid", gap: 18, gridTemplateColumns: "minmax(360px, 1.2fr) minmax(320px, 420px)", minHeight: "100vh", padding: 24, background: "#08131f", color: "#d9e8f4" }}>
            <section style={{ border: "1px solid rgba(150,180,210,0.18)", borderRadius: 24, overflow: "hidden", background: "linear-gradient(180deg, #102033, #08131f)" }}>
                <div style={{ display: "flex", gap: 12, alignItems: "center", justifyContent: "space-between", padding: "18px 20px", borderBottom: "1px solid rgba(150,180,210,0.14)" }}>
                    <div>
                        <div style={{ fontSize: 22, fontWeight: 700 }}>Nanami R1 Preview</div>
                        <div data-testid="nanami-status" style={{ fontSize: 13, opacity: 0.72 }}>{status}</div>
                    </div>
                    <div style={{ display: "flex", gap: 10 }}>
                        <button type="button" onClick={() => void createSession()} style={{ padding: "10px 14px", borderRadius: 999, border: 0, background: "#ff9b45", color: "#08131f", fontWeight: 700 }}>{autoRun ? "Restart Session" : "Create Session"}</button>
                        <button type="button" onClick={() => void loadRobot()} disabled={!sessionId} style={{ padding: "10px 14px", borderRadius: 999, border: "1px solid rgba(255,255,255,0.16)", background: "transparent", color: "#d9e8f4" }}>Load R1</button>
                        <button type="button" onClick={() => void exportFinal()} disabled={!controls.length} style={{ padding: "10px 14px", borderRadius: 999, border: "1px solid rgba(255,255,255,0.16)", background: "transparent", color: "#d9e8f4" }}>Final Render</button>
                        <button type="button" onClick={() => void destroySession()} disabled={!sessionId} style={{ padding: "10px 14px", borderRadius: 999, border: "1px solid rgba(255,255,255,0.16)", background: "transparent", color: "#d9e8f4" }}>Destroy</button>
                    </div>
                </div>
                <div style={{ display: "grid", gap: 8, padding: "16px 20px", borderBottom: "1px solid rgba(150,180,210,0.14)", background: "rgba(4, 12, 20, 0.42)" }}>
                    <div style={{ fontSize: 12, opacity: 0.72 }}>Asset source: local <code>packages/nanami/Temp/URDF/R1</code></div>
                    {autoRun ? <div style={{ fontSize: 12, opacity: 0.72 }}>Auto flow: create, load, update state, export, destroy on unmount.</div> : null}
                </div>
                <div data-testid="nanami-preview" style={{ aspectRatio: "16 / 9", background: "#000" }}>
                    {previewUrl ? <img alt="Nanami preview" src={previewUrl} style={{ width: "100%", height: "100%", objectFit: "cover" }} /> : null}
                </div>
            </section>
            <section style={{ display: "grid", gap: 16 }}>
                <div style={{ border: "1px solid rgba(150,180,210,0.18)", borderRadius: 24, padding: 18, background: "rgba(9, 20, 31, 0.88)" }}>
                    <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 8 }}>Session</div>
                    <div data-testid="nanami-session-id" style={{ fontSize: 12, opacity: 0.72, wordBreak: "break-all" }}>{sessionId || "No active session"}</div>
                </div>
                {controls.map((group) => (
                    <div key={group.name} style={{ border: "1px solid rgba(150,180,210,0.18)", borderRadius: 24, padding: 18, background: "rgba(9, 20, 31, 0.88)" }}>
                        <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 12, textTransform: "capitalize" }}>{group.name.replaceAll("_", " ")}</div>
                        <div style={{ display: "grid", gap: 10 }}>
                            {group.joints.map((joint) => (
                                <label key={joint.name} style={{ display: "grid", gap: 6 }}>
                                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                                        <span>{pretty(joint.name)}</span>
                                        <span>{(values[joint.name] ?? 0).toFixed(joint.kind === "prismatic" ? 3 : 2)}</span>
                                    </div>
                                    <input type="range" min={joint.lower} max={joint.upper} step={joint.kind === "prismatic" ? 0.001 : 0.01} value={values[joint.name] ?? 0} onChange={(event) => setValues((state) => ({ ...state, [joint.name]: Number(event.target.value) }))} />
                                </label>
                            ))}
                        </div>
                    </div>
                ))}
                <div style={{ border: "1px solid rgba(150,180,210,0.18)", borderRadius: 24, padding: 18, background: "rgba(9, 20, 31, 0.88)" }}>
                    <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>Events</div>
                    <div data-testid="nanami-events" style={{ display: "grid", gap: 8, fontSize: 12 }}>
                        {events.map((event, index) => <div key={`${event.type}-${index}`} style={{ opacity: 0.8 }}>{event.type}</div>)}
                    </div>
                </div>
            </section>
        </div>
    )
}
