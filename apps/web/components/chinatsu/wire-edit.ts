import { simplifyPoints } from "./geometry"
import { type Point, type RoutedLink } from "./types"

export type RouteSegment = {
  end: Point
  orientation: "horizontal" | "vertical"
  routeIndex: number
  segmentIndex: number
  start: Point
}

export function getRouteSegments(routes: RoutedLink[]) {
  return routes.flatMap((route) =>
    route.points.slice(1).map((end, segmentIndex) => {
      const start = route.points[segmentIndex]
      return {
        end,
        orientation: start.y === end.y ? "horizontal" : "vertical",
        routeIndex: route.routeIndex,
        segmentIndex,
        start,
      } satisfies RouteSegment
    })
  )
}

export function projectPointToSegment(segment: RouteSegment, point: Point) {
  if (segment.orientation === "horizontal") {
    return {
      x: Math.max(Math.min(point.x, Math.max(segment.start.x, segment.end.x)), Math.min(segment.start.x, segment.end.x)),
      y: segment.start.y,
    }
  }

  return {
    x: segment.start.x,
    y: Math.max(Math.min(point.y, Math.max(segment.start.y, segment.end.y)), Math.min(segment.start.y, segment.end.y)),
  }
}

export function moveRouteSegment(points: Point[], segmentIndex: number, anchor: Point) {
  if (points.length < 2) {
    return points
  }

  const start = points[segmentIndex]
  const end = points[segmentIndex + 1]
  if (!start || !end) {
    return points
  }

  if (start.y === end.y) {
    return simplifyPoints(moveHorizontalSegment(points, segmentIndex, anchor.y))
  }
  return simplifyPoints(moveVerticalSegment(points, segmentIndex, anchor.x))
}

function moveHorizontalSegment(points: Point[], segmentIndex: number, nextY: number) {
  if (segmentIndex === 0) {
    return [points[0], { x: points[0].x, y: nextY }, { x: points[1].x, y: nextY }, ...points.slice(1)]
  }
  if (segmentIndex === points.length - 2) {
    const end = points[points.length - 1]
    const previous = points[points.length - 2]
    return [...points.slice(0, -1), { x: previous.x, y: nextY }, { x: end.x, y: nextY }, end]
  }

  return points.map((point, index) =>
    index === segmentIndex || index === segmentIndex + 1 ? { ...point, y: nextY } : point
  )
}

function moveVerticalSegment(points: Point[], segmentIndex: number, nextX: number) {
  if (segmentIndex === 0) {
    return [points[0], { x: nextX, y: points[0].y }, { x: nextX, y: points[1].y }, ...points.slice(1)]
  }
  if (segmentIndex === points.length - 2) {
    const end = points[points.length - 1]
    const previous = points[points.length - 2]
    return [...points.slice(0, -1), { x: nextX, y: previous.y }, { x: nextX, y: end.y }, end]
  }

  return points.map((point, index) =>
    index === segmentIndex || index === segmentIndex + 1 ? { ...point, x: nextX } : point
  )
}
