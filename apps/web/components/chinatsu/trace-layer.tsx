import { type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent } from "react"

import { collectJointPoints, isPointLinkPin, toPath } from "./geometry"
import { type RouteSegment } from "./wire-edit"
import { type LinkPin, type Point, type Rect, type RoutedLink, CANVAS_HEIGHT, CANVAS_WIDTH } from "./types"

export function CircuitTraceLayer({
  activeSource,
  marqueeRect,
  onBackgroundClick,
  onBackgroundPointerDown,
  onSegmentClick,
  onSegmentPointerDown,
  previewRoute,
  routes,
  segments,
  selectedRouteIndices,
  selectedSegmentKey,
}: {
  activeSource: LinkPin | null
  marqueeRect: Rect | null
  onBackgroundClick: () => void
  onBackgroundPointerDown: (event: ReactPointerEvent<SVGSVGElement>) => void
  onSegmentClick: (segment: RouteSegment, event: ReactMouseEvent<SVGLineElement>) => void
  onSegmentPointerDown: (segment: RouteSegment, event: ReactPointerEvent<SVGLineElement>) => void
  previewRoute: Point[] | null
  routes: RoutedLink[]
  segments: RouteSegment[]
  selectedRouteIndices: number[]
  selectedSegmentKey: string | null
}) {
  const jointPoints = collectJointPoints(routes)
  const activePoint = activeSource && isPointLinkPin(activeSource) ? activeSource : null
  const selectedRouteSet = new Set(selectedRouteIndices)

  return (
    <svg
      width={CANVAS_WIDTH}
      height={CANVAS_HEIGHT}
      onPointerDown={(event) => {
        const target = event.target as Element
        if (target === event.currentTarget || target.getAttribute("data-background") === "true") {
          onBackgroundPointerDown(event)
        }
      }}
      onClick={(event) => {
        const target = event.target as Element
        if (target === event.currentTarget || target.getAttribute("data-background") === "true") {
          onBackgroundClick()
        }
      }}
      style={{ position: "absolute", inset: 0, overflow: "visible" }}
    >
      <rect data-background="true" x={0} y={0} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} fill="transparent" />
      {routes.map((route) => (
        <path key={route.key} d={toPath(route.points)} fill="none" stroke="#b2458f" strokeWidth={2.8} strokeLinejoin="miter" strokeLinecap="square" pointerEvents="none" />
      ))}

      {routes.map((route) => (
        selectedRouteSet.has(route.routeIndex) ? (
          <path key={`selected:${route.key}`} d={toPath(route.points)} fill="none" stroke="#d8922f" strokeWidth={5} strokeLinejoin="miter" strokeLinecap="square" pointerEvents="none" />
        ) : null
      ))}

      {segments.map((segment) => (
        `${segment.routeIndex}:${segment.segmentIndex}` === selectedSegmentKey ? (
          <line
            key={`selected:${segment.routeIndex}:${segment.segmentIndex}`}
            x1={segment.start.x}
            y1={segment.start.y}
            x2={segment.end.x}
            y2={segment.end.y}
            stroke="#d8922f"
            strokeWidth={5}
            strokeLinecap="square"
            pointerEvents="none"
          />
        ) : null
      ))}

      {segments.map((segment) => (
        <line
          key={`${segment.routeIndex}:${segment.segmentIndex}`}
          x1={segment.start.x}
          y1={segment.start.y}
          x2={segment.end.x}
          y2={segment.end.y}
          stroke="transparent"
          strokeWidth={16}
          onClick={(event) => onSegmentClick(segment, event)}
          onPointerDown={(event) => onSegmentPointerDown(segment, event)}
          style={{ cursor: segment.orientation === "horizontal" ? "ns-resize" : "ew-resize" }}
        />
      ))}

      {previewRoute ? <path d={toPath(previewRoute)} fill="none" stroke="#d8922f" strokeWidth={2} strokeDasharray="7 6" strokeLinejoin="miter" strokeLinecap="square" pointerEvents="none" /> : null}
      {marqueeRect ? <rect x={marqueeRect.x} y={marqueeRect.y} width={marqueeRect.width} height={marqueeRect.height} fill="rgba(216, 146, 47, 0.12)" stroke="#d8922f" strokeWidth={2} strokeDasharray="8 6" pointerEvents="none" /> : null}
      {jointPoints.map((point) => (
        <circle key={`${point.x}:${point.y}`} cx={point.x} cy={point.y} r={4} fill="#b2458f" pointerEvents="none" />
      ))}
      {activePoint ? <circle cx={activePoint.x} cy={activePoint.y} r={7} fill="#fff4dc" stroke="#d8922f" strokeWidth={2} pointerEvents="none" /> : null}
    </svg>
  )
}
