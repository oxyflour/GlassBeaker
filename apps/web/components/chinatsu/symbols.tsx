import { type CSSProperties } from "react"

import type { Block, BlockLayout, CircuitBlockType, Point } from "./types"
import { BASE_SPECS, getBlockLayout } from "./symbol-geometry"

const SYMBOL_COLOR = "#305bc9"
const LABEL_COLOR = "#254ba9"

type RenderSpec = {
  height: number
  pins: Point[]
  render: (selected: boolean) => React.ReactNode
  width: number
}

const STROKE = {
  fill: "none",
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  strokeWidth: 2.2,
}

const RENDER_SPECS: Record<CircuitBlockType, RenderSpec> = {
  capacitor: {
    ...BASE_SPECS.capacitor,
    render: (selected) => (
      <g {...STROKE} stroke={selected ? "#b2458f" : SYMBOL_COLOR}>
        <line x1={40} y1={0} x2={40} y2={32} />
        <line x1={24} y1={42} x2={56} y2={42} />
        <line x1={24} y1={54} x2={56} y2={54} />
        <line x1={40} y1={64} x2={40} y2={100} />
      </g>
    ),
  },
  ground: {
    ...BASE_SPECS.ground,
    render: (selected) => (
      <g {...STROKE} stroke={selected ? "#b2458f" : SYMBOL_COLOR}>
        <line x1={20} y1={0} x2={20} y2={18} />
        <line x1={6} y1={20} x2={34} y2={20} />
        <line x1={10} y1={30} x2={30} y2={30} />
        <line x1={14} y1={40} x2={26} y2={40} />
      </g>
    ),
  },
  inductor: {
    ...BASE_SPECS.inductor,
    render: (selected) => (
      <g {...STROKE} stroke={selected ? "#b2458f" : SYMBOL_COLOR}>
        <line x1={0} y1={20} x2={14} y2={20} />
        <path d="M 14 20 C 18 4, 30 4, 34 20 C 38 4, 50 4, 54 20 C 58 4, 70 4, 74 20 C 78 4, 90 4, 94 20" />
        <line x1={94} y1={20} x2={100} y2={20} />
      </g>
    ),
  },
  port: {
    ...BASE_SPECS.port,
    render: (selected) => (
      <g {...STROKE} stroke={selected ? "#b2458f" : SYMBOL_COLOR}>
        <line x1={40} y1={0} x2={40} y2={14} />
        <rect x={12} y={14} width={56} height={72} />
        <line x1={22} y1={28} x2={22} y2={38} />
        <line x1={17} y1={33} x2={27} y2={33} />
        <line x1={40} y1={58} x2={40} y2={70} />
        <line x1={26} y1={70} x2={54} y2={70} />
        <line x1={30} y1={78} x2={50} y2={78} />
      </g>
    ),
  },
  resistor: {
    ...BASE_SPECS.resistor,
    render: (selected) => (
      <g {...STROKE} stroke={selected ? "#b2458f" : SYMBOL_COLOR}>
        <line x1={0} y1={20} x2={16} y2={20} />
        <polyline points="16,20 24,8 32,32 40,8 48,32 56,8 64,32 72,8 80,20" />
        <line x1={80} y1={20} x2={100} y2={20} />
      </g>
    ),
  },
  snp: {
    ...BASE_SPECS.snp,
    render: (selected) => (
      <g {...STROKE} stroke={selected ? "#b2458f" : SYMBOL_COLOR}>
        <line x1={0} y1={30} x2={18} y2={30} />
        <rect x={18} y={10} width={84} height={40} />
        <line x1={102} y1={30} x2={120} y2={30} />
        <text x={60} y={35} fill={selected ? "#b2458f" : SYMBOL_COLOR} fontFamily="Consolas, monospace" fontSize={16} textAnchor="middle">
          SNP
        </text>
      </g>
    ),
  },
}

function getRenderSpec(block: Block): RenderSpec {
  const spec = RENDER_SPECS[block.type ?? "snp"]
  return spec
}

export { getBlockLayout }

export { getBlockRect, getPinPoint } from "./symbol-geometry"

export function getBlockLabelStyle(block: Block): CSSProperties {
  const layout = getBlockLayout(block)
  if (block.rotation === 90 || block.rotation === 270) {
    return { left: Math.max(0, layout.width / 2 - 26), top: layout.height + 8 }
  }
  return { left: layout.width + 10, top: Math.max(0, layout.height / 2 - 16) }
}

export function getBlockLabelColor() {
  return LABEL_COLOR
}

export function CircuitSymbol({ block, selected }: { block: Block; selected: boolean }) {
  const spec = getRenderSpec(block)
  const layout = getBlockLayout(block)
  const transform =
    block.rotation === 90
      ? `translate(${spec.height} 0) rotate(90)`
      : block.rotation === 180
        ? `translate(${spec.width} ${spec.height}) rotate(180)`
        : block.rotation === 270
          ? `translate(0 ${spec.width}) rotate(270)`
          : undefined

  return (
    <svg width={layout.width} height={layout.height} viewBox={`0 0 ${layout.width} ${layout.height}`} style={{ display: "block", overflow: "visible" }}>
      <g transform={transform}>{spec.render(selected)}</g>
    </svg>
  )
}
