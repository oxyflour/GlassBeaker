"use client"

import { useState, useCallback, Suspense, lazy } from "react"
import { Group, Panel, Separator } from "react-resizable-panels"
import type { AntennaOptions, AntennaCut, AntennaNib, EdgePosition } from "../../../../components/nijika/antenna-builder"

const Antenna = lazy(() => import("../../../../components/nijika/antenna"))

const EDGE_POSITIONS: EdgePosition[] = ["top", "bottom", "left", "right"]

function CutEditor({
    cut,
    index,
    onChange,
    onRemove,
}: {
    cut: AntennaCut
    index: number
    onChange: (index: number, cut: AntennaCut) => void
    onRemove: (index: number) => void
}) {
    return (
        <div
            style={{
                display: "flex",
                gap: 8,
                alignItems: "center",
                padding: "8px 12px",
                background: "rgba(7, 17, 28, 0.6)",
                borderRadius: 6,
                border: "1px solid rgba(138, 182, 212, 0.15)",
            }}
        >
            <select
                value={cut.position}
                onChange={(e) => onChange(index, { ...cut, position: e.target.value as EdgePosition })}
                style={{
                    background: "rgba(6, 14, 22, 0.8)",
                    border: "1px solid rgba(138, 182, 212, 0.25)",
                    borderRadius: 4,
                    padding: "4px 8px",
                    color: "#d7e8f4",
                    fontSize: 12,
                }}
            >
                {EDGE_POSITIONS.map((p) => (
                    <option key={p} value={p}>
                        {p}
                    </option>
                ))}
            </select>
            <input
                type="number"
                value={cut.distance.toFixed(2)}
                onChange={(e) => onChange(index, { ...cut, distance: parseFloat(e.target.value) || 0 })}
                step={0.1}
                style={{
                    width: 60,
                    background: "rgba(6, 14, 22, 0.8)",
                    border: "1px solid rgba(138, 182, 212, 0.25)",
                    borderRadius: 4,
                    padding: "4px 8px",
                    color: "#d7e8f4",
                    fontSize: 12,
                }}
            />
            <input
                type="number"
                value={cut.width.toFixed(2)}
                onChange={(e) => onChange(index, { ...cut, width: parseFloat(e.target.value) || 0 })}
                step={0.05}
                style={{
                    width: 50,
                    background: "rgba(6, 14, 22, 0.8)",
                    border: "1px solid rgba(138, 182, 212, 0.25)",
                    borderRadius: 4,
                    padding: "4px 8px",
                    color: "#d7e8f4",
                    fontSize: 12,
                }}
            />
            <button
                onClick={() => onRemove(index)}
                style={{
                    marginLeft: "auto",
                    background: "rgba(255, 100, 100, 0.2)",
                    border: "1px solid rgba(255, 100, 100, 0.3)",
                    borderRadius: 4,
                    padding: "4px 10px",
                    color: "#ff9999",
                    fontSize: 11,
                    cursor: "pointer",
                }}
            >
                x
            </button>
        </div>
    )
}

function NibEditor({
    nib,
    index,
    onChange,
    onRemove,
}: {
    nib: AntennaNib
    index: number
    onChange: (index: number, nib: AntennaNib) => void
    onRemove: (index: number) => void
}) {
    return (
        <div
            style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
                padding: "10px 12px",
                background: "rgba(7, 17, 28, 0.6)",
                borderRadius: 6,
                border: "1px solid rgba(138, 182, 212, 0.15)",
            }}
        >
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <select
                    value={nib.position}
                    onChange={(e) => onChange(index, { ...nib, position: e.target.value as EdgePosition })}
                    style={{
                        background: "rgba(6, 14, 22, 0.8)",
                        border: "1px solid rgba(138, 182, 212, 0.25)",
                        borderRadius: 4,
                        padding: "4px 8px",
                        color: "#d7e8f4",
                        fontSize: 12,
                    }}
                >
                    {EDGE_POSITIONS.map((p) => (
                        <option key={p} value={p}>
                            {p}
                        </option>
                    ))}
                </select>
                <button
                    onClick={() => onRemove(index)}
                    style={{
                        marginLeft: "auto",
                        background: "rgba(255, 100, 100, 0.2)",
                        border: "1px solid rgba(255, 100, 100, 0.3)",
                        borderRadius: 4,
                        padding: "4px 10px",
                        color: "#ff9999",
                        fontSize: 11,
                        cursor: "pointer",
                    }}
                >
                    x
                </button>
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <label style={{ fontSize: 11, color: "#8ab6d4" }}>
                    d:
                    <input
                        type="number"
                        value={nib.distance.toFixed(2)}
                        onChange={(e) => onChange(index, { ...nib, distance: parseFloat(e.target.value) || 0 })}
                        step={0.1}
                        style={{
                            width: 50,
                            marginLeft: 4,
                            background: "rgba(6, 14, 22, 0.8)",
                            border: "1px solid rgba(138, 182, 212, 0.25)",
                            borderRadius: 4,
                            padding: "3px 6px",
                            color: "#d7e8f4",
                            fontSize: 11,
                        }}
                    />
                </label>
                <label style={{ fontSize: 11, color: "#8ab6d4" }}>
                    t:
                    <input
                        type="number"
                        value={nib.thickness.toFixed(2)}
                        onChange={(e) => onChange(index, { ...nib, thickness: parseFloat(e.target.value) || 0 })}
                        step={0.1}
                        style={{
                            width: 45,
                            marginLeft: 4,
                            background: "rgba(6, 14, 22, 0.8)",
                            border: "1px solid rgba(138, 182, 212, 0.25)",
                            borderRadius: 4,
                            padding: "3px 6px",
                            color: "#d7e8f4",
                            fontSize: 11,
                        }}
                    />
                </label>
                <label style={{ fontSize: 11, color: "#8ab6d4" }}>
                    w:
                    <input
                        type="number"
                        value={nib.width.toFixed(2)}
                        onChange={(e) => onChange(index, { ...nib, width: parseFloat(e.target.value) || 0 })}
                        step={0.05}
                        style={{
                            width: 45,
                            marginLeft: 4,
                            background: "rgba(6, 14, 22, 0.8)",
                            border: "1px solid rgba(138, 182, 212, 0.25)",
                            borderRadius: 4,
                            padding: "3px 6px",
                            color: "#d7e8f4",
                            fontSize: 11,
                        }}
                    />
                </label>
            </div>
        </div>
    )
}

function generateDefaultOptions(): AntennaOptions {
    return {
        frame: {
            width: 0.4,
            gap: 1.2,
            cuts: [
                { position: "top", distance: -1.5, width: 0.2 },
                { position: "top", distance: 1.5, width: 0.2 },
                { position: "bottom", distance: -1.5, width: 0.2 },
                { position: "bottom", distance: 1.5, width: 0.2 },
                { position: "left", distance: 1.2, width: 0.2 },
                { position: "right", distance: -1.2, width: 0.2 },
            ],
            nibs: [
                { position: "top", distance: 0, thickness: 1.2, translate: [0, 0, 0], width: 0.3 },
                { position: "bottom", distance: 0, thickness: 1.2, translate: [0, 0, 0], width: 0.3 },
                { position: "left", distance: 1.8, thickness: 1.2, translate: [0, 0, 0], width: 0.3 },
                { position: "right", distance: -1.8, thickness: 1.2, translate: [0, 0, 0], width: 0.3 },
            ],
        },
    }
}

export default function AntennaPreviewPage() {
    const [options, setOptions] = useState<AntennaOptions>(generateDefaultOptions())

    const updateFrame = useCallback((key: keyof AntennaOptions["frame"], value: unknown) => {
        setOptions((prev: AntennaOptions) => ({
            ...prev,
            frame: { ...prev.frame, [key]: value },
        }))
    }, [])

    const updateCut = useCallback((index: number, cut: AntennaCut) => {
        setOptions((prev: AntennaOptions) => ({
            ...prev,
            frame: {
                ...prev.frame,
                cuts: prev.frame.cuts.map((c: AntennaCut, i: number) => (i === index ? cut : c)),
            },
        }))
    }, [])

    const addCut = useCallback(() => {
        setOptions((prev: AntennaOptions) => ({
            ...prev,
            frame: {
                ...prev.frame,
                cuts: [...prev.frame.cuts, { position: "top", distance: 0, width: 0.2 }],
            },
        }))
    }, [])

    const removeCut = useCallback((index: number) => {
        setOptions((prev: AntennaOptions) => ({
            ...prev,
            frame: {
                ...prev.frame,
                cuts: prev.frame.cuts.filter((_: AntennaCut, i: number) => i !== index),
            },
        }))
    }, [])

    const updateNib = useCallback((index: number, nib: AntennaNib) => {
        setOptions((prev: AntennaOptions) => ({
            ...prev,
            frame: {
                ...prev.frame,
                nibs: prev.frame.nibs.map((n: AntennaNib, i: number) => (i === index ? nib : n)),
            },
        }))
    }, [])

    const addNib = useCallback(() => {
        setOptions((prev: AntennaOptions) => ({
            ...prev,
            frame: {
                ...prev.frame,
                nibs: [
                    ...prev.frame.nibs,
                    { position: "top", distance: 0, thickness: 1.2, translate: [0, 0, 0], width: 0.3 },
                ],
            },
        }))
    }, [])

    const removeNib = useCallback((index: number) => {
        setOptions((prev: AntennaOptions) => ({
            ...prev,
            frame: {
                ...prev.frame,
                nibs: prev.frame.nibs.filter((_: AntennaNib, i: number) => i !== index),
            },
        }))
    }, [])

    return (
        <Group orientation="horizontal" style={{ height: "100vh" }}>
            <Panel defaultSize={30} minSize={20} style={{ overflow: "hidden" }}>
                <div
                    style={{
                        height: "100%",
                        background: "linear-gradient(180deg, #0a1929 0%, #07111c 100%)",
                        borderRight: "1px solid rgba(138, 182, 212, 0.15)",
                        padding: "20px",
                        overflow: "auto",
                    }}
                >
                    <h2
                        style={{
                            margin: "0 0 20px 0",
                            fontSize: 16,
                            fontWeight: 600,
                            color: "#d7e8f4",
                            letterSpacing: "0.02em",
                        }}
                    >
                        Antenna Options
                    </h2>

                    {/* Frame Settings */}
                    <div style={{ marginBottom: 24 }}>
                        <h3
                            style={{
                                margin: "0 0 12px 0",
                                fontSize: 12,
                                fontWeight: 500,
                                color: "#8ab6d4",
                                textTransform: "uppercase",
                                letterSpacing: "0.08em",
                            }}
                        >
                            Frame
                        </h3>
                        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                            <label style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                                <span style={{ fontSize: 13, color: "#a8c5dc" }}>Width</span>
                                <input
                                    type="number"
                                    value={options.frame.width.toFixed(2)}
                                    onChange={(e) => updateFrame("width", parseFloat(e.target.value) || 0)}
                                    step={0.05}
                                    style={{
                                        width: 70,
                                        background: "rgba(6, 14, 22, 0.8)",
                                        border: "1px solid rgba(138, 182, 212, 0.25)",
                                        borderRadius: 4,
                                        padding: "6px 10px",
                                        color: "#d7e8f4",
                                        fontSize: 13,
                                    }}
                                />
                            </label>
                            <label style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                                <span style={{ fontSize: 13, color: "#a8c5dc" }}>Gap</span>
                                <input
                                    type="number"
                                    value={options.frame.gap.toFixed(2)}
                                    onChange={(e) => updateFrame("gap", parseFloat(e.target.value) || 0)}
                                    step={0.1}
                                    style={{
                                        width: 70,
                                        background: "rgba(6, 14, 22, 0.8)",
                                        border: "1px solid rgba(138, 182, 212, 0.25)",
                                        borderRadius: 4,
                                        padding: "6px 10px",
                                        color: "#d7e8f4",
                                        fontSize: 13,
                                    }}
                                />
                            </label>
                        </div>
                    </div>

                    {/* Cuts */}
                    <div style={{ marginBottom: 24 }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                            <h3
                                style={{
                                    margin: 0,
                                    fontSize: 12,
                                    fontWeight: 500,
                                    color: "#8ab6d4",
                                    textTransform: "uppercase",
                                    letterSpacing: "0.08em",
                                }}
                            >
                                Cuts ({options.frame.cuts.length})
                            </h3>
                            <button
                                onClick={addCut}
                                style={{
                                    background: "rgba(73, 211, 214, 0.15)",
                                    border: "1px solid rgba(73, 211, 214, 0.3)",
                                    borderRadius: 4,
                                    padding: "4px 12px",
                                    color: "#49d3d6",
                                    fontSize: 11,
                                    cursor: "pointer",
                                }}
                            >
                                + Add
                            </button>
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                            {options.frame.cuts.map((cut: AntennaCut, i: number) => (
                                <CutEditor key={i} cut={cut} index={i} onChange={updateCut} onRemove={removeCut} />
                            ))}
                        </div>
                    </div>

                    {/* Nibs */}
                    <div>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                            <h3
                                style={{
                                    margin: 0,
                                    fontSize: 12,
                                    fontWeight: 500,
                                    color: "#8ab6d4",
                                    textTransform: "uppercase",
                                    letterSpacing: "0.08em",
                                }}
                            >
                                Nibs ({options.frame.nibs.length})
                            </h3>
                            <button
                                onClick={addNib}
                                style={{
                                    background: "rgba(73, 211, 214, 0.15)",
                                    border: "1px solid rgba(73, 211, 214, 0.3)",
                                    borderRadius: 4,
                                    padding: "4px 12px",
                                    color: "#49d3d6",
                                    fontSize: 11,
                                    cursor: "pointer",
                                }}
                            >
                                + Add
                            </button>
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                            {options.frame.nibs.map((nib: AntennaNib, i: number) => (
                                <NibEditor key={i} nib={nib} index={i} onChange={updateNib} onRemove={removeNib} />
                            ))}
                        </div>
                    </div>
                </div>
            </Panel>

            <Separator />

            <Panel defaultSize={70} minSize={30}>
                <Suspense fallback={null}>
                    <Antenna options={options} />
                </Suspense>
            </Panel>
        </Group>
    )
}
