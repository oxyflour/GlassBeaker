import { clamp, snap } from "./geometry"
import { getBlockLayout } from "./symbols"
import { type Block, type Point, CANVAS_HEIGHT, CANVAS_WIDTH, GRID_SIZE } from "./types"

export function findNearestLegalPosition(block: Block, target: Point) {
  const layout = getBlockLayout(block)
  return {
    x: clamp(snap(target.x, GRID_SIZE), 0, CANVAS_WIDTH - layout.width),
    y: clamp(snap(target.y, GRID_SIZE), 0, CANVAS_HEIGHT - layout.height),
  }
}
