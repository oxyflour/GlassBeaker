import type { Block, BlockLayout, CircuitBlockType, Point, Rect } from "./types"

type BaseSpec = {
  height: number
  pins: Point[]
  width: number
}

const BASE_SPECS: Record<CircuitBlockType, BaseSpec> = {
  capacitor: {
    width: 80,
    height: 100,
    pins: [{ x: 40, y: 0 }, { x: 40, y: 100 }],
  },
  ground: {
    width: 40,
    height: 60,
    pins: [{ x: 20, y: 0 }],
  },
  inductor: {
    width: 100,
    height: 40,
    pins: [{ x: 0, y: 20 }, { x: 100, y: 20 }],
  },
  port: {
    width: 80,
    height: 100,
    pins: [{ x: 40, y: 0 }],
  },
  resistor: {
    width: 100,
    height: 40,
    pins: [{ x: 0, y: 20 }, { x: 100, y: 20 }],
  },
  snp: {
    width: 120,
    height: 60,
    pins: [{ x: 0, y: 30 }, { x: 120, y: 30 }],
  },
}

function rotatePoint(point: Point, width: number, height: number, rotation: Block["rotation"] = 0): Point {
  switch (rotation) {
    case 90:
      return { x: height - point.y, y: point.x }
    case 180:
      return { x: width - point.x, y: height - point.y }
    case 270:
      return { x: point.y, y: width - point.x }
    default:
      return point
  }
}

export function getBaseSpec(block: Block) {
  return BASE_SPECS[block.type ?? "snp"]
}

export function getBlockLayout(block: Block): BlockLayout {
  const spec = getBaseSpec(block)
  const rotated = block.rotation === 90 || block.rotation === 270
  return {
    width: rotated ? spec.height : spec.width,
    height: rotated ? spec.width : spec.height,
    pinPoints: spec.pins.map((pin) => rotatePoint(pin, spec.width, spec.height, block.rotation)),
  }
}

export function getBlockRect(block: Block, position: Point, padding = 0): Rect {
  const layout = getBlockLayout(block)
  return {
    x: position.x - padding,
    y: position.y - padding,
    width: layout.width + padding * 2,
    height: layout.height + padding * 2,
  }
}

export function getPinPoint(block: Block, position: Point, pin: number) {
  const layout = getBlockLayout(block)
  const safePin = layout.pinPoints[Math.max(0, Math.min(pin, layout.pinPoints.length - 1))]
  return { x: position.x + safePin.x, y: position.y + safePin.y }
}

export { BASE_SPECS }
