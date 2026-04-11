import { clamp } from "./geometry"
import { CANVAS_HEIGHT, CANVAS_WIDTH, type Point } from "./types"

export type ViewportSize = {
  height: number
  width: number
}

export type ViewportState = {
  scale: number
  x: number
  y: number
}

function clampAxis(value: number, contentSize: number, viewportSize: number) {
  if (viewportSize <= 0) {
    return value
  }
  if (contentSize <= viewportSize) {
    return clamp(value, 0, viewportSize - contentSize)
  }
  return clamp(value, viewportSize - contentSize, 0)
}

export function clampViewport(view: ViewportState, size: ViewportSize): ViewportState {
  const contentWidth = CANVAS_WIDTH * view.scale
  const contentHeight = CANVAS_HEIGHT * view.scale
  return {
    ...view,
    x: clampAxis(view.x, contentWidth, size.width),
    y: clampAxis(view.y, contentHeight, size.height),
  }
}

export function zoomViewport(view: ViewportState, size: ViewportSize, anchor: Point, nextScale: number) {
  const canvasX = (anchor.x - view.x) / view.scale
  const canvasY = (anchor.y - view.y) / view.scale
  return clampViewport(
    {
      scale: nextScale,
      x: anchor.x - canvasX * nextScale,
      y: anchor.y - canvasY * nextScale,
    },
    size
  )
}
