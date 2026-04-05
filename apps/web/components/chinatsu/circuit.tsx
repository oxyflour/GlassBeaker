"use client"

import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react"

export interface BlockPin {
    name?: string
}

export interface Block {
    id: string
    pins?: BlockPin[]
}

export interface LinkPin {
    node: string
    pin: number
}

export interface Link {
    from: LinkPin
    to: LinkPin
}

export interface CircuitData {
    blocks: Block[]
    links: Link[]
}

type Point = {
    x: number
    y: number
}

type PositionMap = Record<string, Point>

type DragState = {
    id: string
    offsetX: number
    offsetY: number
}

type Route = {
    key: string
    points: Point[]
    link: Link
}

const GRID_SIZE = 24
const BLOCK_WIDTH = 176
const BLOCK_HEADER_HEIGHT = 36
const PIN_ROW_HEIGHT = 30
const BLOCK_PADDING = 12
const CANVAS_WIDTH = 1200
const CANVAS_HEIGHT = 720

const DEFAULT_CIRCUIT: CircuitData = {
    blocks: [
        {
            id: "Clock",
            pins: [{ name: "tick" }],
        },
        {
            id: "Sequencer",
            pins: [{ name: "clk" }, { name: "gate" }, { name: "step" }],
        },
        {
            id: "Mixer",
            pins: [{ name: "a" }, { name: "b" }, { name: "mix" }],
        },
        {
            id: "Output",
            pins: [{ name: "signal" }],
        },
    ],
    links: [
        {
            from: { node: "Clock", pin: 0 },
            to: { node: "Sequencer", pin: 0 },
        },
        {
            from: { node: "Sequencer", pin: 2 },
            to: { node: "Mixer", pin: 0 },
        },
        {
            from: { node: "Clock", pin: 0 },
            to: { node: "Mixer", pin: 1 },
        },
        {
            from: { node: "Mixer", pin: 2 },
            to: { node: "Output", pin: 0 },
        },
    ],
}

function snap(value: number) {
    return Math.round(value / GRID_SIZE) * GRID_SIZE
}

function clamp(value: number, min: number, max: number) {
    return Math.max(min, Math.min(max, value))
}

function getDefaultPosition(index: number): Point {
    const column = index % 4
    const row = Math.floor(index / 4)
    return {
        x: GRID_SIZE * (2 + column * 10),
        y: GRID_SIZE * (2 + row * 8),
    }
}

function syncPositions(blocks: Block[], previous: PositionMap): PositionMap {
    const next: PositionMap = {}
    blocks.forEach((block, index) => {
        next[block.id] = previous[block.id] ?? getDefaultPosition(index)
    })
    return next
}

function getRowCount(block: Block) {
    return Math.max(block.pins?.length ?? 0, 1)
}

function getBlockHeight(block: Block) {
    return BLOCK_HEADER_HEIGHT + BLOCK_PADDING * 2 + getRowCount(block) * PIN_ROW_HEIGHT
}

function getPinY(block: Block, pin: number) {
    const rowCount = getRowCount(block)
    const row = clamp(pin, 0, rowCount - 1)
    return BLOCK_HEADER_HEIGHT + BLOCK_PADDING + row * PIN_ROW_HEIGHT + PIN_ROW_HEIGHT / 2
}

function getPinPoint(block: Block, position: Point, pin: number, side: "left" | "right"): Point {
    return {
        x: position.x + (side === "left" ? 0 : BLOCK_WIDTH),
        y: position.y + getPinY(block, pin),
    }
}

function buildRoute(from: Point, to: Point): Point[] {
    const forward = to.x - from.x >= GRID_SIZE * 4
    const laneX = forward
        ? snap((from.x + to.x) / 2)
        : snap(Math.max(from.x, to.x) + GRID_SIZE * 3)

    const points = [
        from,
        { x: laneX, y: from.y },
        { x: laneX, y: to.y },
        to,
    ]

    return points.filter((point, index) => {
        if (index === 0) {
            return true
        }
        const previous = points[index - 1]
        return previous.x !== point.x || previous.y !== point.y
    })
}

function toPath(points: Point[]) {
    return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ")
}

function pointKey(point: Point) {
    return `${point.x}:${point.y}`
}

function getSegments(points: Point[]) {
    return points.slice(1).map((point, index) => ({
        start: points[index],
        end: point,
    }))
}

function isBetween(value: number, start: number, end: number) {
    return value >= Math.min(start, end) && value <= Math.max(start, end)
}

function collectJointPoints(routes: Route[]) {
    const counts = new Map<string, { count: number, point: Point }>()

    function increment(point: Point) {
        const key = pointKey(point)
        const previous = counts.get(key)
        counts.set(key, {
            count: (previous?.count ?? 0) + 1,
            point,
        })
    }

    routes.forEach((route) => {
        route.points.slice(1, -1).forEach((point) => {
            increment(point)
        })
    })

    for (let routeIndex = 0; routeIndex < routes.length; routeIndex += 1) {
        const segmentsA = getSegments(routes[routeIndex].points)
        for (let otherIndex = routeIndex + 1; otherIndex < routes.length; otherIndex += 1) {
            const segmentsB = getSegments(routes[otherIndex].points)
            segmentsA.forEach((segmentA) => {
                segmentsB.forEach((segmentB) => {
                    const horizontal =
                        segmentA.start.y === segmentA.end.y && segmentB.start.x === segmentB.end.x
                            ? { h: segmentA, v: segmentB }
                            : segmentA.start.x === segmentA.end.x && segmentB.start.y === segmentB.end.y
                                ? { h: segmentB, v: segmentA }
                                : null

                    if (!horizontal) {
                        return
                    }

                    const point = {
                        x: horizontal.v.start.x,
                        y: horizontal.h.start.y,
                    }

                    if (
                        isBetween(point.x, horizontal.h.start.x, horizontal.h.end.x) &&
                        isBetween(point.y, horizontal.v.start.y, horizontal.v.end.y)
                    ) {
                        increment(point)
                    }
                })
            })
        }
    }

    return Array.from(counts.values())
        .filter((entry) => entry.count > 1)
        .map((entry) => entry.point)
}

function samePin(a: LinkPin, b: LinkPin) {
    return a.node === b.node && a.pin === b.pin
}

function comparePins(a: LinkPin, b: LinkPin, positions: PositionMap) {
    const aPosition = positions[a.node]
    const bPosition = positions[b.node]

    if (aPosition && bPosition) {
        if (aPosition.x !== bPosition.x) {
            return aPosition.x - bPosition.x
        }
        if (aPosition.y !== bPosition.y) {
            return aPosition.y - bPosition.y
        }
    }

    if (a.node !== b.node) {
        return a.node.localeCompare(b.node)
    }

    return a.pin - b.pin
}

export default function Circuit({
    data: controlledData,
    onChange,
}: {
    data?: CircuitData
    onChange?: (data: CircuitData) => void
}) {
    const [localData, setLocalData] = useState<CircuitData>(() => controlledData ?? DEFAULT_CIRCUIT)
    const [positions, setPositions] = useState<PositionMap>(() =>
        syncPositions((controlledData ?? DEFAULT_CIRCUIT).blocks, {})
    )
    const [dragging, setDragging] = useState<DragState | null>(null)
    const [activeSource, setActiveSource] = useState<LinkPin | null>(null)
    const [cursorPoint, setCursorPoint] = useState<Point | null>(null)
    const canvasRef = useRef<HTMLDivElement | null>(null)

    const data = controlledData ?? localData

    useEffect(() => {
        setPositions((previous) => syncPositions(data.blocks, previous))
    }, [data.blocks])

    useEffect(() => {
        if (!dragging) {
            return
        }

        const dragState = dragging

        function handlePointerMove(event: PointerEvent) {
            const rect = canvasRef.current?.getBoundingClientRect()
            if (!rect) {
                return
            }

            const nextX = clamp(snap(event.clientX - rect.left - dragState.offsetX), 0, CANVAS_WIDTH - BLOCK_WIDTH)
            const draggedBlock = data.blocks.find((block) => block.id === dragState.id)
            const maxY = CANVAS_HEIGHT - (draggedBlock ? getBlockHeight(draggedBlock) : BLOCK_HEADER_HEIGHT)
            const nextY = clamp(snap(event.clientY - rect.top - dragState.offsetY), 0, maxY)

            setPositions((previous) => ({
                ...previous,
                [dragState.id]: { x: nextX, y: nextY },
            }))
        }

        function handlePointerUp() {
            setDragging(null)
        }

        window.addEventListener("pointermove", handlePointerMove)
        window.addEventListener("pointerup", handlePointerUp)
        return () => {
            window.removeEventListener("pointermove", handlePointerMove)
            window.removeEventListener("pointerup", handlePointerUp)
        }
    }, [data.blocks, dragging])

    const blockMap = useMemo(
        () => Object.fromEntries(data.blocks.map((block) => [block.id, block])),
        [data.blocks]
    )

    const routes = useMemo<Route[]>(() => {
        return data.links.flatMap((link, index) => {
            const sourceBlock = blockMap[link.from.node]
            const targetBlock = blockMap[link.to.node]
            const sourcePosition = positions[link.from.node]
            const targetPosition = positions[link.to.node]

            if (!sourceBlock || !targetBlock || !sourcePosition || !targetPosition) {
                return []
            }

            return [{
                key: `${link.from.node}:${link.from.pin}->${link.to.node}:${link.to.pin}:${index}`,
                link,
                points: buildRoute(
                    getPinPoint(sourceBlock, sourcePosition, link.from.pin, "right"),
                    getPinPoint(targetBlock, targetPosition, link.to.pin, "left")
                ),
            }]
        })
    }, [blockMap, data.links, positions])

    const jointPoints = useMemo(() => collectJointPoints(routes), [routes])

    const previewRoute = useMemo(() => {
        if (!activeSource || !cursorPoint) {
            return null
        }

        const sourceBlock = blockMap[activeSource.node]
        const sourcePosition = positions[activeSource.node]
        if (!sourceBlock || !sourcePosition) {
            return null
        }

        return buildRoute(getPinPoint(sourceBlock, sourcePosition, activeSource.pin, "right"), cursorPoint)
    }, [activeSource, blockMap, cursorPoint, positions])

    function emitChange(next: CircuitData) {
        if (!controlledData) {
            setLocalData(next)
        }
        onChange?.(next)
    }

    function beginDrag(id: string, event: ReactPointerEvent<HTMLDivElement>) {
        const rect = canvasRef.current?.getBoundingClientRect()
        const position = positions[id]
        if (!rect || !position) {
            return
        }

        event.preventDefault()
        setDragging({
            id,
            offsetX: event.clientX - rect.left - position.x,
            offsetY: event.clientY - rect.top - position.y,
        })
    }

    function updateCursor(event: ReactPointerEvent<HTMLDivElement>) {
        if (!activeSource) {
            return
        }

        const rect = canvasRef.current?.getBoundingClientRect()
        if (!rect) {
            return
        }

        setCursorPoint({
            x: clamp(snap(event.clientX - rect.left), 0, CANVAS_WIDTH),
            y: clamp(snap(event.clientY - rect.top), 0, CANVAS_HEIGHT),
        })
    }

    function clearActivePin() {
        setActiveSource(null)
        setCursorPoint(null)
    }

    function handlePinClick(target: LinkPin) {
        if (!activeSource) {
            const sourceBlock = blockMap[target.node]
            const sourcePosition = positions[target.node]

            setActiveSource(target)
            setCursorPoint(
                sourceBlock && sourcePosition
                    ? getPinPoint(sourceBlock, sourcePosition, target.pin, "right")
                    : null
            )
            return
        }

        if (samePin(activeSource, target)) {
            clearActivePin()
            return
        }

        const [from, to] = comparePins(activeSource, target, positions) <= 0
            ? [activeSource, target]
            : [target, activeSource]

        const duplicate = data.links.some((link) =>
            (samePin(link.from, from) && samePin(link.to, to)) ||
            (samePin(link.from, to) && samePin(link.to, from))
        )

        if (!duplicate) {
            emitChange({
                ...data,
                links: [...data.links, { from, to }],
            })
        }

        clearActivePin()
    }

    function removeLink(target: Link) {
        emitChange({
            ...data,
            links: data.links.filter((link) => !(
                samePin(link.from, target.from) &&
                samePin(link.to, target.to)
            )),
        })
    }

    return (
        <div
            ref={canvasRef}
            onPointerMove={updateCursor}
            onPointerLeave={() => setCursorPoint(null)}
            onClick={(event) => {
                if (event.target === canvasRef.current) {
                    clearActivePin()
                }
            }}
            style={{
                position: "relative",
                width: "100%",
                minHeight: CANVAS_HEIGHT,
                overflow: "auto",
                borderRadius: 20,
                border: "1px solid #223244",
                backgroundColor: "#08121d",
                backgroundImage: `
                    linear-gradient(rgba(92, 128, 167, 0.18) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(92, 128, 167, 0.18) 1px, transparent 1px)
                `,
                backgroundSize: `${GRID_SIZE}px ${GRID_SIZE}px`,
                boxShadow: "inset 0 1px 0 rgba(255, 255, 255, 0.04)",
            }}
        >
            <div style={{ position: "relative", width: CANVAS_WIDTH, height: CANVAS_HEIGHT }}>
                <svg
                    width={CANVAS_WIDTH}
                    height={CANVAS_HEIGHT}
                    style={{ position: "absolute", inset: 0, overflow: "visible" }}
                >
                    {routes.map((route) => (
                        <g key={route.key}>
                            <path
                                d={toPath(route.points)}
                                fill="none"
                                stroke="#78d4ff"
                                strokeWidth={3}
                                strokeLinejoin="round"
                                strokeLinecap="round"
                            />
                            <path
                                d={toPath(route.points)}
                                fill="none"
                                stroke="transparent"
                                strokeWidth={16}
                                strokeLinecap="round"
                                onClick={() => removeLink(route.link)}
                                style={{ cursor: "pointer" }}
                            />
                        </g>
                    ))}
                    {previewRoute ? (
                        <path
                            d={toPath(previewRoute)}
                            fill="none"
                            stroke="#ffd166"
                            strokeWidth={2}
                            strokeDasharray="8 6"
                            strokeLinejoin="round"
                            strokeLinecap="round"
                        />
                    ) : null}
                    {jointPoints.map((point) => (
                        <circle
                            key={pointKey(point)}
                            cx={point.x}
                            cy={point.y}
                            r={4.5}
                            fill="#08121d"
                            stroke="#78d4ff"
                            strokeWidth={2}
                        />
                    ))}
                </svg>

                {data.blocks.map((block) => {
                    const position = positions[block.id] ?? getDefaultPosition(0)
                    const pins = block.pins ?? []
                    const blockHeight = getBlockHeight(block)

                    return (
                        <div
                            key={block.id}
                            style={{
                                position: "absolute",
                                left: position.x,
                                top: position.y,
                                width: BLOCK_WIDTH,
                                minHeight: blockHeight,
                                borderRadius: 16,
                                border: "1px solid #31485f",
                                background: "linear-gradient(180deg, rgba(17, 28, 39, 0.96), rgba(8, 18, 29, 0.96))",
                                boxShadow: "0 14px 28px rgba(0, 0, 0, 0.28)",
                                userSelect: "none",
                            }}
                        >
                            <div
                                onPointerDown={(event) => beginDrag(block.id, event)}
                                style={{
                                    display: "flex",
                                    alignItems: "center",
                                    height: BLOCK_HEADER_HEIGHT,
                                    padding: "0 12px",
                                    borderBottom: "1px solid #223244",
                                    cursor: dragging?.id === block.id ? "grabbing" : "grab",
                                    color: "#f0f4f8",
                                    fontWeight: 700,
                                    letterSpacing: "0.02em",
                                }}
                            >
                                <span>{block.id}</span>
                            </div>
                            <div style={{ padding: `${BLOCK_PADDING}px 10px` }}>
                                {(pins.length ? pins : [{ name: "Pin 1" }]).map((pin, index) => {
                                    const sourceSelected = activeSource?.node === block.id && activeSource.pin === index

                                    return (
                                        <button
                                            key={`${block.id}:${index}`}
                                            type="button"
                                            onClick={() => handlePinClick({ node: block.id, pin: index })}
                                            style={{
                                                display: "grid",
                                                gridTemplateColumns: "20px 1fr 20px",
                                                alignItems: "center",
                                                gap: 8,
                                                width: "100%",
                                                height: PIN_ROW_HEIGHT,
                                                padding: 0,
                                                border: 0,
                                                borderRadius: 8,
                                                background: sourceSelected ? "rgba(255, 209, 102, 0.14)" : "transparent",
                                                color: "#c7d2db",
                                                cursor: "pointer",
                                                textAlign: "left",
                                            }}
                                            aria-label={`Connect ${block.id} ${pin.name ?? `pin ${index + 1}`}`}
                                        >
                                            <span
                                                style={{
                                                    width: 14,
                                                    height: 14,
                                                    borderRadius: 999,
                                                    border: `2px solid ${sourceSelected ? "#ffd166" : "#78d4ff"}`,
                                                    background: sourceSelected ? "#4b3800" : "#0d2233",
                                                    justifySelf: "start",
                                                    boxSizing: "border-box",
                                                }}
                                            />
                                            <span
                                                style={{
                                                    overflow: "hidden",
                                                    textOverflow: "ellipsis",
                                                    whiteSpace: "nowrap",
                                                    fontSize: 13,
                                                }}
                                            >
                                                {pin.name ?? `Pin ${index + 1}`}
                                            </span>
                                            <span
                                                style={{
                                                    width: 14,
                                                    height: 14,
                                                    borderRadius: 999,
                                                    border: `2px solid ${sourceSelected ? "#ffd166" : "#78d4ff"}`,
                                                    background: sourceSelected ? "#4b3800" : "#0d2233",
                                                    justifySelf: "end",
                                                    boxSizing: "border-box",
                                                }}
                                            />
                                        </button>
                                    )
                                })}
                            </div>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
