import { getSegments, rectsIntersect, segmentIntersectsRect } from "./geometry"
import { getBlockRect } from "./symbols"
import { type Block, type Point, type Rect, type RoutedLink } from "./types"

export function toSelectionRect(start: Point, end: Point): Rect {
  return {
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  }
}

export function findBlocksInRect(blocks: Block[], positions: Record<string, Point>, rect: Rect) {
  return blocks
    .filter((block) => {
      const position = positions[block.id] ?? block.position
      return position ? rectsIntersect(getBlockRect(block, position, 8), rect) : false
    })
    .map((block) => block.id)
}

export function findRoutesInRect(routes: RoutedLink[], rect: Rect) {
  return routes
    .filter((route) =>
      getSegments(route.points).some((segment) => segmentIntersectsRect(segment.start, segment.end, rect))
    )
    .map((route) => route.routeIndex)
}
