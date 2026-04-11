"use client"

import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent, type WheelEvent as ReactWheelEvent } from "react"

import { CircuitBlockView } from "./block-view"
import { DEFAULT_CIRCUIT, createBlock, normalizeCircuitData, syncPositions } from "./defaults"
import { clamp, comparePins, isPointLinkPin, isPointOnSegment, samePin } from "./geometry"
import { findNearestLegalPosition } from "./placement"
import { buildPreviewRoute } from "./routing"
import { useRoutingWorker } from "./use-routing-worker"
import { findBlocksInRect, findRoutesInRect, toSelectionRect } from "./selection"
import { getBlockLayout } from "./symbols"
import { CircuitTraceLayer } from "./trace-layer"
import { type Block, type BlockPin, type CircuitBlockType, type CircuitData, type Link, type LinkPin, type Point, type RoutedLink, CANVAS_HEIGHT, CANVAS_WIDTH, GRID_SIZE } from "./types"
import { clampViewport, type ViewportSize, type ViewportState, zoomViewport } from "./viewport"
import { getRouteSegments, moveRouteSegment, projectPointToSegment, type RouteSegment } from "./wire-edit"

export type { Block, BlockPin, CircuitBlockType, CircuitData, Link, LinkPin } from "./types"

type DragState = { id: string; offsetX: number; offsetY: number }
type SegmentDragState = {
  axis: number
  moved: boolean
  originalAxis: number
  originalPoints: Point[]
  routeIndex: number
  segment: RouteSegment
}
type SelectedSegmentState = {
  from: LinkPin
  routeIndex: number
  segmentIndex: number
  to: LinkPin
}
type CanvasInteraction =
  | {
      kind: "marquee"
      current: Point
      moved: boolean
      origin: Point
      startClientX: number
      startClientY: number
    }
  | {
      kind: "pan"
      moved: boolean
      startClientX: number
      startClientY: number
      startView: ViewportState
    }

type ContextMenuState = {
  x: number
  y: number
  canvasPoint: Point
  blockId?: string
} | null

type EditBlockState = {
  block: Block
} | null

const MIN_SCALE = 0.5
const MAX_SCALE = 2.5
const WHEEL_ZOOM_STEP = 1.12
const DRAG_THRESHOLD = 4

function withLinkOverrides(data: CircuitData, overrides: Record<number, Point[]>) {
  if (Object.keys(overrides).length === 0) {
    return data
  }
  return {
    ...data,
    links: data.links.map((link, index) => overrides[index] ? { ...link, points: overrides[index] } : link),
  }
}

function shiftPointEndpoint(pin: LinkPin, start: Point, end: Point, delta: Point): LinkPin {
  if (!isPointLinkPin(pin) || !isPointOnSegment(pin, start, end)) {
    return pin
  }
  return { ...pin, x: pin.x + delta.x, y: pin.y + delta.y }
}

function isEditableTarget(target: EventTarget | null) {
  return target instanceof HTMLElement && (target.isContentEditable || ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName))
}

function sameNumberSet(left: number[], right: number[]) {
  return left.length === right.length && left.every((value, index) => value === right[index])
}

function sameStringSet(left: string[], right: string[]) {
  return left.length === right.length && left.every((value, index) => value === right[index])
}

function uniqueNumbers(values: number[]) {
  return Array.from(new Set(values)).sort((left, right) => left - right)
}

function uniqueStrings(values: string[]) {
  return Array.from(new Set(values))
}


function zoomStep(scale: number, direction: "in" | "out") {
  const multiplier = direction === "in" ? WHEEL_ZOOM_STEP : 1 / WHEEL_ZOOM_STEP
  return clamp(Number((scale * multiplier).toFixed(3)), MIN_SCALE, MAX_SCALE)
}

export default function Circuit({ data: controlledData, onChange }: { data?: CircuitData; onChange?: (data: CircuitData) => void }) {
  const [localData, setLocalData] = useState(() => normalizeCircuitData(controlledData ?? DEFAULT_CIRCUIT))
  const data = useMemo(() => normalizeCircuitData(controlledData ?? localData), [controlledData, localData])
  const [positions, setPositions] = useState(() => syncPositions(data.blocks, {}))
  const [selectedBlockIds, setSelectedBlockIds] = useState<string[]>(() => data.blocks[0]?.id ? [data.blocks[0].id] : [])
  const [selectedRouteIndices, setSelectedRouteIndices] = useState<number[]>([])
  const [selectedSegment, setSelectedSegment] = useState<SelectedSegmentState | null>(null)
  const [dragging, setDragging] = useState<DragState | null>(null)
  const [segmentDragging, setSegmentDragging] = useState<SegmentDragState | null>(null)
  const [activeSource, setActiveSource] = useState<LinkPin | null>(null)
  const [cursorPoint, setCursorPoint] = useState<Point | null>(null)
  const [linkOverrides, setLinkOverrides] = useState<Record<number, Point[]>>({})
  const [view, setView] = useState<ViewportState>({ scale: 1, x: 0, y: 0 })
  const [viewportSize, setViewportSize] = useState<ViewportSize>({ height: CANVAS_HEIGHT, width: CANVAS_WIDTH })
  const [canvasInteraction, setCanvasInteraction] = useState<CanvasInteraction | null>(null)
  const [spacePressed, setSpacePressed] = useState(false)
  const [contextMenu, setContextMenu] = useState<ContextMenuState>(null)
  const [editingBlock, setEditingBlock] = useState<EditBlockState>(null)
  const viewportRef = useRef<HTMLDivElement | null>(null)
  const latestRef = useRef({ data, positions, routes: [] as RoutedLink[] })
  const overridesRef = useRef(linkOverrides)
  const segmentDraggingRef = useRef<SegmentDragState | null>(segmentDragging)
  const viewRef = useRef(view)
  const viewportSizeRef = useRef(viewportSize)
  const canvasInteractionRef = useRef<CanvasInteraction | null>(canvasInteraction)
  const suppressTraceClickRef = useRef(false)
  const suppressBackgroundClickRef = useRef(false)

  useEffect(() => {
    setPositions((previous) => syncPositions(data.blocks, previous))
  }, [data.blocks])

  useEffect(() => {
    viewRef.current = view
  }, [view])

  useEffect(() => {
    viewportSizeRef.current = viewportSize
  }, [viewportSize])

  useEffect(() => {
    canvasInteractionRef.current = canvasInteraction
  }, [canvasInteraction])

  useEffect(() => {
    setSelectedBlockIds((previous) => {
      const valid = previous.filter((id) => data.blocks.some((block) => block.id === id))
      return sameStringSet(previous, valid) ? previous : valid
    })
  }, [data.blocks])

  useEffect(() => {
    setSelectedRouteIndices((previous) => {
      const valid = uniqueNumbers(previous.filter((index) => index >= 0 && index < data.links.length))
      return sameNumberSet(previous, valid) ? previous : valid
    })
  }, [data.links.length])

  const visualData = useMemo(() => withLinkOverrides(data, linkOverrides), [data, linkOverrides])
  const { routes } = useRoutingWorker(visualData, positions)
  const segments = useMemo(() => getRouteSegments(routes), [routes])
  const blockMap = useMemo(() => Object.fromEntries(data.blocks.map((block) => [block.id, block])), [data.blocks])
  const selectedSegmentKey = selectedSegment ? `${selectedSegment.routeIndex}:${selectedSegment.segmentIndex}` : null
  const highlightedRouteIndices = useMemo(
    () => uniqueNumbers(selectedSegment ? [...selectedRouteIndices, selectedSegment.routeIndex] : selectedRouteIndices),
    [selectedRouteIndices, selectedSegment]
  )
  const previewRoute = useMemo(
    () => (activeSource && cursorPoint ? buildPreviewRoute(activeSource, cursorPoint, data.blocks, positions, routes) : null),
    [activeSource, cursorPoint, data.blocks, positions, routes]
  )
  const marqueeRect = canvasInteraction?.kind === "marquee" && canvasInteraction.moved ? toSelectionRect(canvasInteraction.origin, canvasInteraction.current) : null

  useEffect(() => {
    latestRef.current = { data, positions, routes }
    overridesRef.current = linkOverrides
    segmentDraggingRef.current = segmentDragging
  }, [data, positions, routes, linkOverrides, segmentDragging])

  useEffect(() => {
    if (!selectedSegment) {
      return
    }

    const link = data.links[selectedSegment.routeIndex]
    const segmentExists = segments.some((segment) => segment.routeIndex === selectedSegment.routeIndex && segment.segmentIndex === selectedSegment.segmentIndex)
    if (!link || !segmentExists || !samePin(link.from, selectedSegment.from) || !samePin(link.to, selectedSegment.to)) {
      setSelectedSegment(null)
    }
  }, [data.links, segments, selectedSegment])

  useEffect(() => {
    const node = viewportRef.current
    if (!node) {
      return
    }

    const updateSize = () => {
      const next = { height: node.clientHeight, width: node.clientWidth }
      viewportSizeRef.current = next
      setViewportSize((previous) => previous.height === next.height && previous.width === next.width ? previous : next)
      setView((previous) => {
        const clamped = clampViewport(previous, next)
        return clamped.scale === previous.scale && clamped.x === previous.x && clamped.y === previous.y ? previous : clamped
      })
    }

    updateSize()
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", updateSize)
      return () => window.removeEventListener("resize", updateSize)
    }

    const observer = new ResizeObserver(updateSize)
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.code === "Space" && !isEditableTarget(event.target)) {
        event.preventDefault()
        setSpacePressed(true)
        return
      }

      if ((event.key !== "Backspace" && event.key !== "Delete") || event.altKey || event.ctrlKey || event.metaKey || isEditableTarget(event.target)) {
        return
      }
      if (selectedBlockIds.length === 0 && highlightedRouteIndices.length === 0) {
        return
      }
      if (dragging || segmentDragging || canvasInteraction) {
        return
      }

      event.preventDefault()
      deleteSelection()
    }

    function handleKeyUp(event: KeyboardEvent) {
      if (event.code === "Space") {
        setSpacePressed(false)
      }
    }

    function handleBlur() {
      setSpacePressed(false)
    }

    window.addEventListener("keydown", handleKeyDown)
    window.addEventListener("keyup", handleKeyUp)
    window.addEventListener("blur", handleBlur)
    return () => {
      window.removeEventListener("keydown", handleKeyDown)
      window.removeEventListener("keyup", handleKeyUp)
      window.removeEventListener("blur", handleBlur)
    }
  }, [canvasInteraction, deleteSelection, dragging, highlightedRouteIndices, segmentDragging, selectedBlockIds])

  useEffect(() => {
    if (!dragging) {
      return
    }

    const dragState = dragging
    function handlePointerMove(event: PointerEvent) {
      const block = latestRef.current.data.blocks.find((item) => item.id === dragState.id)
      const point = toCanvasPoint(event.clientX, event.clientY, false)
      if (!block || !point) {
        return
      }
      setPositions((previous) => ({
        ...previous,
        [dragState.id]: findNearestLegalPosition(block, {
          x: clamp(Math.round((point.x - dragState.offsetX) / GRID_SIZE) * GRID_SIZE, 0, CANVAS_WIDTH),
          y: clamp(Math.round((point.y - dragState.offsetY) / GRID_SIZE) * GRID_SIZE, 0, CANVAS_HEIGHT),
        }),
      }))
    }

    function handlePointerUp() {
      const next = latestRef.current.positions[dragState.id]
      if (next) {
        emitChange({
          ...latestRef.current.data,
          blocks: latestRef.current.data.blocks.map((block) => block.id === dragState.id ? { ...block, position: next } : block),
        })
      }
      setDragging(null)
    }

    window.addEventListener("pointermove", handlePointerMove)
    window.addEventListener("pointerup", handlePointerUp)
    return () => {
      window.removeEventListener("pointermove", handlePointerMove)
      window.removeEventListener("pointerup", handlePointerUp)
    }
  }, [dragging])

  useEffect(() => {
    if (!segmentDragging) {
      return
    }

    const dragState = segmentDragging
    function handlePointerMove(event: PointerEvent) {
      const point = toCanvasPoint(event.clientX, event.clientY)
      if (!point) {
        return
      }

      const anchor = dragState.segment.orientation === "horizontal"
        ? { x: dragState.segment.start.x, y: point.y }
        : { x: point.x, y: dragState.segment.start.y }
      const axis = dragState.segment.orientation === "horizontal" ? point.y : point.x
      suppressTraceClickRef.current ||= axis !== dragState.originalAxis
      setLinkOverrides((previous) => ({ ...previous, [dragState.routeIndex]: moveRouteSegment(dragState.originalPoints, dragState.segment.segmentIndex, anchor) }))
      setSegmentDragging((previous) => previous ? { ...previous, axis, moved: previous.moved || axis !== previous.originalAxis } : previous)
    }

    function handlePointerUp() {
      const latestDrag = segmentDraggingRef.current
      const override = overridesRef.current[dragState.routeIndex]
      if (latestDrag?.moved && override) {
        const delta = latestDrag.segment.orientation === "horizontal"
          ? { x: 0, y: latestDrag.axis - latestDrag.originalAxis }
          : { x: latestDrag.axis - latestDrag.originalAxis, y: 0 }

        emitChange({
          ...latestRef.current.data,
          links: latestRef.current.data.links.map((link, index) => {
            const shifted = {
              ...link,
              from: shiftPointEndpoint(link.from, latestDrag.segment.start, latestDrag.segment.end, delta),
              to: shiftPointEndpoint(link.to, latestDrag.segment.start, latestDrag.segment.end, delta),
            }
            return index === dragState.routeIndex ? { ...shifted, points: override } : shifted
          }),
        })
      }

      setLinkOverrides((previous) => Object.fromEntries(Object.entries(previous).filter(([key]) => Number(key) !== dragState.routeIndex)))
      setSegmentDragging(null)
    }

    window.addEventListener("pointermove", handlePointerMove)
    window.addEventListener("pointerup", handlePointerUp)
    return () => {
      window.removeEventListener("pointermove", handlePointerMove)
      window.removeEventListener("pointerup", handlePointerUp)
    }
  }, [segmentDragging])

  useEffect(() => {
    if (!canvasInteraction) {
      return
    }

    function handlePointerMove(event: PointerEvent) {
      const current = canvasInteractionRef.current
      if (!current) {
        return
      }

      if (current.kind === "pan") {
        const dx = event.clientX - current.startClientX
        const dy = event.clientY - current.startClientY
        const moved = current.moved || Math.abs(dx) > DRAG_THRESHOLD || Math.abs(dy) > DRAG_THRESHOLD
        setView(clampViewport({ ...current.startView, x: current.startView.x + dx, y: current.startView.y + dy }, viewportSizeRef.current))
        if (moved !== current.moved) {
          const next = { ...current, moved }
          canvasInteractionRef.current = next
          setCanvasInteraction(next)
        }
        return
      }

      const point = toCanvasPoint(event.clientX, event.clientY, false)
      if (!point) {
        return
      }
      const moved = current.moved || Math.abs(event.clientX - current.startClientX) > DRAG_THRESHOLD || Math.abs(event.clientY - current.startClientY) > DRAG_THRESHOLD
      const next = moved === current.moved && point.x === current.current.x && point.y === current.current.y ? current : { ...current, current: point, moved }
      if (next !== current) {
        canvasInteractionRef.current = next
        setCanvasInteraction(next)
      }
    }

    function handlePointerUp() {
      const current = canvasInteractionRef.current
      if (current?.moved) {
        suppressBackgroundClickRef.current = true
      }
      if (current?.kind === "marquee" && current.moved) {
        const rect = toSelectionRect(current.origin, current.current)
        replaceSelection(findBlocksInRect(latestRef.current.data.blocks, latestRef.current.positions, rect), findRoutesInRect(latestRef.current.routes, rect))
      }
      canvasInteractionRef.current = null
      setCanvasInteraction(null)
    }

    window.addEventListener("pointermove", handlePointerMove)
    window.addEventListener("pointerup", handlePointerUp)
    return () => {
      window.removeEventListener("pointermove", handlePointerMove)
      window.removeEventListener("pointerup", handlePointerUp)
    }
  }, [canvasInteraction])

  function emitChange(next: CircuitData) {
    const normalized = normalizeCircuitData(next)
    if (!controlledData) {
      setLocalData(normalized)
    }
    onChange?.(normalized)
  }

  function toCanvasPoint(clientX: number, clientY: number, snapToGrid = true) {
    const rect = viewportRef.current?.getBoundingClientRect()
    if (!rect) {
      return null
    }

    const currentView = viewRef.current
    const rawX = clamp((clientX - rect.left - currentView.x) / currentView.scale, 0, CANVAS_WIDTH)
    const rawY = clamp((clientY - rect.top - currentView.y) / currentView.scale, 0, CANVAS_HEIGHT)
    return snapToGrid
      ? {
          x: clamp(Math.round(rawX / GRID_SIZE) * GRID_SIZE, 0, CANVAS_WIDTH),
          y: clamp(Math.round(rawY / GRID_SIZE) * GRID_SIZE, 0, CANVAS_HEIGHT),
        }
      : { x: rawX, y: rawY }
  }

  function replaceSelection(blockIds: string[], routeIndices: number[], segment: SelectedSegmentState | null = null) {
    setSelectedBlockIds(uniqueStrings(blockIds))
    setSelectedRouteIndices(uniqueNumbers(routeIndices))
    setSelectedSegment(segment)
  }

  function clearPendingLink() {
    setActiveSource(null)
    setCursorPoint(null)
  }

  function clearSelection() {
    replaceSelection([], [])
  }

  function selectBlock(id: string) {
    replaceSelection([id], [])
  }

  function selectSegment(segment: RouteSegment) {
    const link = data.links[segment.routeIndex]
    if (!link) {
      return
    }
    replaceSelection([], [segment.routeIndex], { from: link.from, routeIndex: segment.routeIndex, segmentIndex: segment.segmentIndex, to: link.to })
  }

  function handleEndpointClick(target: LinkPin) {
    setSelectedRouteIndices([])
    setSelectedSegment(null)
    if (!activeSource) {
      setActiveSource(target)
      setCursorPoint(isPointLinkPin(target) ? target : null)
      return
    }
    if (samePin(activeSource, target)) {
      clearPendingLink()
      return
    }

    const [from, to] = comparePins(activeSource, target, positions) <= 0 ? [activeSource, target] : [target, activeSource]
    const duplicate = data.links.some((link) => (samePin(link.from, from) && samePin(link.to, to)) || (samePin(link.from, to) && samePin(link.to, from)))
    if (!duplicate) {
      emitChange({ ...data, links: [...data.links, { from, to }] })
    }
    clearPendingLink()
  }

  function beginDrag(id: string, event: ReactPointerEvent<HTMLDivElement>) {
    if (spacePressed || activeSource) {
      return
    }

    const position = positions[id]
    const point = toCanvasPoint(event.clientX, event.clientY, false)
    if (!position || !point) {
      return
    }

    event.stopPropagation()
    selectBlock(id)
    setDragging({
      id,
      offsetX: clamp(point.x - position.x, 0, getBlockLayout(blockMap[id] ?? { id, type: "snp" }).width),
      offsetY: clamp(point.y - position.y, 0, getBlockLayout(blockMap[id] ?? { id, type: "snp" }).height),
    })
  }

  function deleteSelection() {
    const blockIds = new Set(selectedBlockIds)
    const routeIndices = new Set(highlightedRouteIndices)
    if (blockIds.size === 0 && routeIndices.size === 0) {
      return
    }

    const nextBlocks = data.blocks.filter((block) => !blockIds.has(block.id))
    setPositions((previous) => Object.fromEntries(Object.entries(previous).filter(([id]) => !blockIds.has(id))))
    clearSelection()
    clearPendingLink()
    emitChange({
      blocks: nextBlocks,
      links: data.links.filter((link, index) => {
        if (routeIndices.has(index)) {
          return false
        }
        return !(!isPointLinkPin(link.from) && blockIds.has(link.from.node)) && !(!isPointLinkPin(link.to) && blockIds.has(link.to.node))
      }),
    })
  }

  function startSegmentDrag(segment: RouteSegment, event: React.PointerEvent<SVGLineElement>) {
    if (activeSource) {
      return
    }

    const route = routes.find((item) => item.routeIndex === segment.routeIndex)
    if (!route) {
      return
    }

    event.stopPropagation()
    suppressTraceClickRef.current = false
    selectSegment(segment)
    setSegmentDragging({
      axis: segment.orientation === "horizontal" ? segment.start.y : segment.start.x,
      moved: false,
      originalAxis: segment.orientation === "horizontal" ? segment.start.y : segment.start.x,
      originalPoints: route.points,
      routeIndex: segment.routeIndex,
      segment: segment,
    })
  }

  function handleTraceClick(segment: RouteSegment, event: React.MouseEvent<SVGLineElement>) {
    if (suppressTraceClickRef.current) {
      suppressTraceClickRef.current = false
      return
    }

    event.stopPropagation()
    if (!activeSource) {
      clearPendingLink()
      selectSegment(segment)
      return
    }

    const point = toCanvasPoint(event.clientX, event.clientY)
    if (!point) {
      return
    }
    handleEndpointClick({ kind: "point", ...projectPointToSegment(segment, point) })
  }

  function beginCanvasInteraction(event: ReactPointerEvent<HTMLElement | SVGSVGElement>, allowMarquee: boolean) {
    const wantsPan = event.button === 1 || event.button === 2 || spacePressed
    if (!wantsPan && (!allowMarquee || event.button !== 0 || activeSource || dragging || segmentDragging)) {
      return
    }
    if (wantsPan && (dragging || segmentDragging)) {
      return
    }

    event.preventDefault()
    if (wantsPan) {
      const next = {
        kind: "pan" as const,
        moved: false,
        startClientX: event.clientX,
        startClientY: event.clientY,
        startView: viewRef.current,
      }
      canvasInteractionRef.current = next
      setCanvasInteraction(next)
      return
    }

    const point = toCanvasPoint(event.clientX, event.clientY, false)
    if (!point) {
      return
    }
    const next = {
      kind: "marquee" as const,
      current: point,
      moved: false,
      origin: point,
      startClientX: event.clientX,
      startClientY: event.clientY,
    }
    canvasInteractionRef.current = next
    setCanvasInteraction(next)
  }

  function handleBackgroundClick() {
    if (suppressBackgroundClickRef.current) {
      suppressBackgroundClickRef.current = false
      return
    }
    clearSelection()
    clearPendingLink()
  }

  function panByWheel(event: ReactWheelEvent<HTMLDivElement>) {
    event.preventDefault()
    const rect = viewportRef.current?.getBoundingClientRect()
    if (!rect) {
      return
    }
    const anchor = { x: event.clientX - rect.left, y: event.clientY - rect.top }
    setView((previous) => zoomViewport(previous, viewportSizeRef.current, anchor, zoomStep(previous.scale, event.deltaY < 0 ? "in" : "out")))
  }

  function findBlockAtPoint(point: Point): string | undefined {
    for (const block of data.blocks) {
      const pos = positions[block.id] ?? block.position
      if (!pos) continue
      const layout = getBlockLayout(block)
      if (
        point.x >= pos.x &&
        point.x <= pos.x + layout.width &&
        point.y >= pos.y &&
        point.y <= pos.y + layout.height
      ) {
        return block.id
      }
    }
    return undefined
  }

  function handleContextMenu(event: React.MouseEvent) {
    event.preventDefault()
    const point = toCanvasPoint(event.clientX, event.clientY)
    if (!point) return
    const blockId = findBlockAtPoint(point)
    setContextMenu({ x: event.clientX, y: event.clientY, canvasPoint: point, blockId })
  }

  function closeContextMenu() {
    setContextMenu(null)
  }

  function hasSelection() {
    return selectedBlockIds.length > 0 || highlightedRouteIndices.length > 0
  }

  function addBlockAtPosition(type: CircuitBlockType, position: Point) {
    const candidate = createBlock(type, data.blocks, position)
    const legalPosition = findNearestLegalPosition(candidate, position)
    const nextBlock = { ...candidate, position: legalPosition }
    setPositions((previous) => ({ ...previous, [nextBlock.id]: legalPosition }))
    selectBlock(nextBlock.id)
    emitChange({ ...data, blocks: [...data.blocks, nextBlock] })
    closeContextMenu()
  }

  return (
    <div style={{ border: "1px solid #1b2433", background: "#161c29", overflow: "hidden", userSelect: "none", position: "relative" }}>
      <div
        ref={viewportRef}
        onPointerMove={(event) => {
          if (!activeSource) {
            return
          }
          const point = toCanvasPoint(event.clientX, event.clientY)
          if (point) {
            setCursorPoint(point)
          }
        }}
        onPointerDown={(event) => {
          if (event.target === event.currentTarget) {
            beginCanvasInteraction(event, false)
          }
        }}
        onClick={(event) => {
          if (event.target === event.currentTarget) {
            handleBackgroundClick()
            closeContextMenu()
          }
        }}
        onContextMenu={handleContextMenu}
        onWheel={panByWheel}
        style={{
          cursor: canvasInteraction?.kind === "pan" ? "grabbing" : spacePressed ? "grab" : "default",
          height: CANVAS_HEIGHT,
          maxWidth: "100%",
          overflow: "hidden",
          position: "relative",
          touchAction: "none",
        }}
      >
        <div
          style={{
            height: CANVAS_HEIGHT,
            left: 0,
            position: "absolute",
            top: 0,
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
            transformOrigin: "top left",
            width: CANVAS_WIDTH,
          }}
        >
          <div
            style={{
              backgroundColor: "#efe8dc",
              backgroundImage: "radial-gradient(circle at 1px 1px, rgba(72, 89, 130, 0.28) 1px, transparent 0)",
              backgroundSize: `${GRID_SIZE}px ${GRID_SIZE}px`,
              height: CANVAS_HEIGHT,
              position: "relative",
              userSelect: "none",
              width: CANVAS_WIDTH,
            }}
          >
            <CircuitTraceLayer
              activeSource={activeSource}
              marqueeRect={marqueeRect}
              onBackgroundClick={handleBackgroundClick}
              onBackgroundPointerDown={(event) => beginCanvasInteraction(event, true)}
              onSegmentClick={handleTraceClick}
              onSegmentPointerDown={startSegmentDrag}
              previewRoute={previewRoute}
              routes={routes}
              segments={segments}
              selectedRouteIndices={highlightedRouteIndices}
              selectedSegmentKey={selectedSegmentKey}
            />

            {data.blocks.map((block) => (
              <CircuitBlockView
                key={block.id}
                activeSource={activeSource}
                block={block}
                draggingId={dragging?.id ?? null}
                onBeginDrag={beginDrag}
                onPinClick={handleEndpointClick}
                onSelect={selectBlock}
                position={positions[block.id] ?? block.position ?? { x: 0, y: 0 }}
                selected={selectedBlockIds.includes(block.id)}
              />
            ))}
          </div>
        </div>

        {contextMenu && (
          <div
            onClick={closeContextMenu}
            style={{
              position: "fixed",
              inset: 0,
              zIndex: 100,
            }}
          >
            <div
              onClick={(e) => e.stopPropagation()}
              style={{
                position: "absolute",
                left: contextMenu.x,
                top: contextMenu.y,
                background: "#1e2636",
                border: "1px solid #2d3a52",
                borderRadius: 4,
                boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
                padding: "4px 0",
                minWidth: 120,
                zIndex: 101,
              }}
            >
              {(contextMenu.blockId || selectedBlockIds.length === 1) && (
                <>
                  <button
                    onClick={() => {
                      const targetId = contextMenu.blockId || selectedBlockIds[0]
                      const block = data.blocks.find((b) => b.id === targetId)
                      if (block) {
                        setEditingBlock({ block })
                      }
                      closeContextMenu()
                    }}
                    style={{
                      display: "block",
                      width: "100%",
                      padding: "6px 12px",
                      textAlign: "left",
                      background: "transparent",
                      border: "none",
                      color: "#4dabf7",
                      fontSize: 13,
                      cursor: "pointer",
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = "#2d3a52")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                  >
                    编辑器件
                  </button>
                  <div style={{ margin: "4px 0", borderBottom: "1px solid #2d3a52" }} />
                </>
              )}
              {hasSelection() && (
                <>
                  <button
                    onClick={() => {
                      deleteSelection()
                      closeContextMenu()
                    }}
                    style={{
                      display: "block",
                      width: "100%",
                      padding: "6px 12px",
                      textAlign: "left",
                      background: "transparent",
                      border: "none",
                      color: "#ff6b6b",
                      fontSize: 13,
                      cursor: "pointer",
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = "#2d3a52")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                  >
                    删除选中
                  </button>
                  <div style={{ margin: "4px 0", borderBottom: "1px solid #2d3a52" }} />
                </>
              )}
              <div style={{ padding: "4px 12px", color: "#8b9bb4", fontSize: 12 }}>
                添加器件
              </div>
              {([
                { type: "snp" as const, label: "S参数文件" },
                { type: "resistor" as const, label: "电阻" },
                { type: "capacitor" as const, label: "电容" },
                { type: "inductor" as const, label: "电感" },
                { type: "ground" as const, label: "接地" },
              ]).map(({ type, label }) => (
                <button
                  key={type}
                  onClick={() => addBlockAtPosition(type, contextMenu.canvasPoint)}
                  style={{
                    display: "block",
                    width: "100%",
                    padding: "6px 12px",
                    textAlign: "left",
                    background: "transparent",
                    border: "none",
                    color: "#b8c0d4",
                    fontSize: 13,
                    cursor: "pointer",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "#2d3a52")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        )}

        {editingBlock && (
          <div
            onClick={() => setEditingBlock(null)}
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(0,0,0,0.5)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 200,
            }}
          >
            <div
              onClick={(e) => e.stopPropagation()}
              style={{
                background: "#1e2636",
                border: "1px solid #2d3a52",
                borderRadius: 8,
                padding: 20,
                minWidth: 280,
              }}
            >
              <div style={{ color: "#e8ecf5", fontSize: 16, fontWeight: 500, marginBottom: 16 }}>
                编辑器件
              </div>
              <div style={{ marginBottom: 12 }}>
                <label style={{ display: "block", color: "#8b9bb4", fontSize: 12, marginBottom: 6 }}>
                  标签
                </label>
                <input
                  id="block-label-input"
                  type="text"
                  defaultValue={editingBlock.block.label ?? ""}
                  style={{
                    width: "100%",
                    padding: "8px 12px",
                    background: "#161c29",
                    border: "1px solid #2d3a52",
                    borderRadius: 4,
                    color: "#e8ecf5",
                    fontSize: 14,
                    outline: "none",
                    boxSizing: "border-box",
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      const labelInput = document.getElementById("block-label-input") as HTMLInputElement
                      const valueInput = document.getElementById("block-value-input") as HTMLInputElement
                      updateBlockValue(editingBlock.block.id, labelInput?.value ?? "", valueInput?.value ?? "")
                      setEditingBlock(null)
                    }
                  }}
                />
              </div>
              <div style={{ marginBottom: 20 }}>
                <label style={{ display: "block", color: "#8b9bb4", fontSize: 12, marginBottom: 6 }}>
                  值
                </label>
                <textarea
                  id="block-value-input"
                  defaultValue={editingBlock.block.value ?? ""}
                  rows={3}
                  style={{
                    width: "100%",
                    padding: "8px 12px",
                    background: "#161c29",
                    border: "1px solid #2d3a52",
                    borderRadius: 4,
                    color: "#e8ecf5",
                    fontSize: 14,
                    outline: "none",
                    resize: "vertical",
                    boxSizing: "border-box",
                    fontFamily: "inherit",
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && e.ctrlKey) {
                      const labelInput = document.getElementById("block-label-input") as HTMLInputElement
                      const valueInput = document.getElementById("block-value-input") as HTMLInputElement
                      updateBlockValue(editingBlock.block.id, labelInput?.value ?? "", valueInput?.value ?? "")
                      setEditingBlock(null)
                    }
                  }}
                />
              </div>
              <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
                <button
                  onClick={() => setEditingBlock(null)}
                  style={{
                    padding: "6px 16px",
                    background: "transparent",
                    border: "1px solid #2d3a52",
                    borderRadius: 4,
                    color: "#8b9bb4",
                    fontSize: 13,
                    cursor: "pointer",
                  }}
                >
                  取消
                </button>
                <button
                  onClick={() => {
                    const labelInput = document.getElementById("block-label-input") as HTMLInputElement
                    const valueInput = document.getElementById("block-value-input") as HTMLInputElement
                    updateBlockValue(editingBlock.block.id, labelInput?.value ?? "", valueInput?.value ?? "")
                    setEditingBlock(null)
                  }}
                  style={{
                    padding: "6px 16px",
                    background: "#4dabf7",
                    border: "none",
                    borderRadius: 4,
                    color: "#fff",
                    fontSize: 13,
                    cursor: "pointer",
                  }}
                >
                  确定
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )

  function updateBlockValue(blockId: string, label: string, value: string) {
    emitChange({
      ...data,
      blocks: data.blocks.map((block) =>
        block.id === blockId ? { ...block, label: label || undefined, value: value || undefined } : block
      ),
    })
  }
}
