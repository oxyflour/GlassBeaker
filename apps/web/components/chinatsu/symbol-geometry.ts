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

function extractPortCountFromSnp(value: string | undefined): number {
  if (!value) return 2
  const match = value.match(/\bs(\d+)p\b/i)
  if (match) {
    const count = Number.parseInt(match[1], 10)
    return count >= 1 && count <= 99 ? count : 2
  }
  return 2
}

function calculateSnpHeight(portCount: number): number {
  if (portCount > 6) {
    return Math.max(60, portCount * 12)
  } else if (portCount > 4) {
    return 80
  }
  return 60
}

function generateSnpPins(portCount: number, height: number): Point[] {
  const pins: Point[] = []
  const width = 120

  if (portCount <= 2) {
    pins.push({ x: 0, y: height / 2 })
    pins.push({ x: width, y: height / 2 })
  } else if (portCount <= 4) {
    const leftY = height / 3
    const rightY = (height * 2) / 3
    pins.push({ x: 0, y: leftY })
    pins.push({ x: width, y: leftY })
    pins.push({ x: 0, y: rightY })
    pins.push({ x: width, y: rightY })
  } else if (portCount <= 6) {
    const leftY1 = height / 4
    const rightY1 = height / 4
    const leftY2 = height / 2
    const rightY2 = height / 2
    const leftY3 = (height * 3) / 4
    const rightY3 = (height * 3) / 4
    pins.push({ x: 0, y: leftY1 })
    pins.push({ x: width, y: rightY1 })
    pins.push({ x: 0, y: leftY2 })
    pins.push({ x: width, y: rightY2 })
    pins.push({ x: 0, y: leftY3 })
    pins.push({ x: width, y: rightY3 })
  } else {
    const maxPortsPerSide = Math.ceil(portCount / 2)
    for (let i = 0; i < maxPortsPerSide; i++) {
      const y = ((i + 1) * height) / (maxPortsPerSide + 1)
      pins.push({ x: 0, y })
      if (pins.length < portCount) {
        pins.push({ x: width, y })
      }
    }
  }
  return pins.slice(0, portCount)
}

function generateSnpPinNames(portCount: number): string[] {
  return Array.from({ length: portCount }, (_, i) => `P${i + 1}`)
}

function getSnpSpec(block: Block): BaseSpec {
  const portCount = extractPortCountFromSnp(block.value)
  const height = calculateSnpHeight(portCount)
  const pins = generateSnpPins(portCount, height)
  return { width: 120, height, pins }
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

export function getBaseSpec(block: Block): BaseSpec {
  if (block.type === "snp") {
    return getSnpSpec(block)
  }
  return BASE_SPECS[block.type ?? "snp"]
}

export { generateSnpPinNames }

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
export { extractPortCountFromSnp }
