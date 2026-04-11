import { type LinkPin, type Point, type Rect, type RoutedLink } from "./types"

export function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

export function snap(value: number, size: number) {
  return Math.round(value / size) * size
}

export function pointKey(point: Point) {
  return `${point.x}:${point.y}`
}

export function isPointLinkPin(pin: LinkPin): pin is Extract<LinkPin, { kind: "point" }> {
  return pin.kind === "point"
}

export function samePin(a: LinkPin, b: LinkPin) {
  if (isPointLinkPin(a) || isPointLinkPin(b)) {
    return isPointLinkPin(a) && isPointLinkPin(b) && a.x === b.x && a.y === b.y
  }
  return a.node === b.node && a.pin === b.pin
}

export function comparePins(a: LinkPin, b: LinkPin, positions: Record<string, Point>) {
  const left = isPointLinkPin(a) ? a : positions[a.node]
  const right = isPointLinkPin(b) ? b : positions[b.node]
  if (left && right && left.x !== right.x) {
    return left.x - right.x
  }
  if (left && right && left.y !== right.y) {
    return left.y - right.y
  }
  if (isPointLinkPin(a) || isPointLinkPin(b)) {
    return pointKey(isPointLinkPin(a) ? a : left ?? { x: 0, y: 0 }).localeCompare(pointKey(isPointLinkPin(b) ? b : right ?? { x: 0, y: 0 }))
  }
  return a.node === b.node ? a.pin - b.pin : a.node.localeCompare(b.node)
}

export function toPath(points: Point[]) {
  return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ")
}

export function simplifyPoints(points: Point[]) {
  return points.filter((point, index, source) => {
    if (index === 0 || index === source.length - 1) {
      return true
    }

    const previous = source[index - 1]
    const next = source[index + 1]
    return !((previous.x === point.x && point.x === next.x) || (previous.y === point.y && point.y === next.y))
  })
}

export function rectsIntersect(a: Rect, b: Rect) {
  return a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y
}

export function segmentIntersectsRect(start: Point, end: Point, rect: Rect) {
  if (start.y === end.y) {
    const y = start.y
    const left = Math.min(start.x, end.x)
    const right = Math.max(start.x, end.x)
    return y >= rect.y && y <= rect.y + rect.height && right >= rect.x && left <= rect.x + rect.width
  }

  const x = start.x
  const top = Math.min(start.y, end.y)
  const bottom = Math.max(start.y, end.y)
  return x >= rect.x && x <= rect.x + rect.width && bottom >= rect.y && top <= rect.y + rect.height
}

export function getSegments(points: Point[]) {
  return points.slice(1).map((point, index) => ({ end: point, start: points[index] }))
}

export function isPointOnSegment(point: Point, start: Point, end: Point) {
  if (start.x === end.x) {
    return point.x === start.x && isBetween(point.y, start.y, end.y)
  }
  if (start.y === end.y) {
    return point.y === start.y && isBetween(point.x, start.x, end.x)
  }
  return false
}

function isBetween(value: number, start: number, end: number) {
  return value >= Math.min(start, end) && value <= Math.max(start, end)
}

export function collectJointPoints(routes: RoutedLink[]) {
  const counts = new Map<string, { count: number; point: Point }>()
  const mark = (point: Point) => {
    const key = pointKey(point)
    const previous = counts.get(key)
    counts.set(key, { count: (previous?.count ?? 0) + 1, point })
  }

  routes.forEach((route) => route.points.slice(1, -1).forEach(mark))
  for (let left = 0; left < routes.length; left += 1) {
    for (let right = left + 1; right < routes.length; right += 1) {
      getSegments(routes[left].points).forEach((first) => {
        getSegments(routes[right].points).forEach((second) => {
          const pair =
            first.start.y === first.end.y && second.start.x === second.end.x
              ? { h: first, v: second }
              : first.start.x === first.end.x && second.start.y === second.end.y
                ? { h: second, v: first }
                : null
          if (!pair) {
            return
          }

          const point = { x: pair.v.start.x, y: pair.h.start.y }
          if (
            isBetween(point.x, pair.h.start.x, pair.h.end.x) &&
            isBetween(point.y, pair.v.start.y, pair.v.end.y)
          ) {
            mark(point)
          }
        })
      })
    }
  }

  return Array.from(counts.values())
    .filter((entry) => entry.count > 1)
    .map((entry) => entry.point)
}
