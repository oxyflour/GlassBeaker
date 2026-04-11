import { generateSnpPinNames } from "./symbol-geometry"
import { type Block, type CircuitBlockType, type CircuitData, type PaletteItem, type Point, GRID_SIZE } from "./types"

type BlockMeta = {
  label: string
  pinNames: string[]
  prefix: string
  rotation: 0 | 90 | 180 | 270
  value?: string
  valueFactory?: (index: number) => string | undefined
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

export const BLOCK_META: Record<CircuitBlockType, BlockMeta> = {
  capacitor: { prefix: "C", label: "C", pinNames: ["top", "bottom"], rotation: 0, value: "C=1 pF {t}" },
  ground: { prefix: "GND", label: "", pinNames: ["net"], rotation: 0 },
  inductor: { prefix: "L", label: "L", pinNames: ["left", "right"], rotation: 0, value: "L=1 nH {t}" },
  port: {
    prefix: "TermG",
    label: "TermG",
    pinNames: ["net"],
    rotation: 0,
    valueFactory: (index) => `Num=${index}\nZ=50 Ohm`,
  },
  resistor: { prefix: "R", label: "R", pinNames: ["left", "right"], rotation: 0, value: "R=50 Ohm {t}" },
  snp: { prefix: "SNP", label: "SNP", pinNames: ["in", "out"], rotation: 0, value: "file=network.s2p" },
}

export const PALETTE_ITEMS: PaletteItem[] = [
  { type: "port", label: "Port", hint: "50 Ohm terminal" },
  { type: "capacitor", label: "C", hint: "Shunt or series capacitor" },
  { type: "inductor", label: "L", hint: "Series or shunt inductor" },
  { type: "resistor", label: "R", hint: "Series or shunt resistor" },
  { type: "snp", label: "SNP", hint: "Touchstone network block" },
  { type: "ground", label: "GND", hint: "Ground reference" },
]

const DEFAULT_CIRCUIT_SEED: CircuitData = {
  blocks: [
    { id: "TermG1", label: "TermG", position: { x: 120, y: 140 }, type: "port" },
    { id: "C1", position: { x: 300, y: 140 }, type: "capacitor" },
    { id: "L1", position: { x: 480, y: 120 }, type: "inductor" },
    { id: "C2", position: { x: 760, y: 140 }, type: "capacitor" },
    { id: "TermG2", label: "TermG", position: { x: 980, y: 140 }, type: "port" },
    { id: "GND1", position: { x: 320, y: 300 }, type: "ground" },
    { id: "GND2", position: { x: 780, y: 300 }, type: "ground" },
  ],
  links: [
    { from: { node: "TermG1", pin: 0 }, to: { node: "C1", pin: 0 } },
    { from: { node: "C1", pin: 0 }, to: { node: "L1", pin: 0 } },
    { from: { node: "L1", pin: 1 }, to: { node: "C2", pin: 0 } },
    { from: { node: "C2", pin: 0 }, to: { node: "TermG2", pin: 0 } },
    { from: { node: "C1", pin: 1 }, to: { node: "GND1", pin: 0 } },
    { from: { node: "C2", pin: 1 }, to: { node: "GND2", pin: 0 } },
  ],
}

export const DEFAULT_CIRCUIT = normalizeCircuitData(DEFAULT_CIRCUIT_SEED)

function buildPins(pinNames: string[]) {
  return pinNames.map((name) => ({ name }))
}

function getDefaultPosition(index: number): Point {
  const column = index % 5
  const row = Math.floor(index / 5)
  return {
    x: GRID_SIZE * (5 + column * 10),
    y: GRID_SIZE * (6 + row * 8),
  }
}

function nextOrdinal(type: CircuitBlockType, blocks: Block[]) {
  const prefix = BLOCK_META[type].prefix
  return blocks.reduce((max, block) => {
    const match = block.id.match(new RegExp(`^${prefix}(\\d+)$`))
    return Math.max(max, match ? Number(match[1]) : 0)
  }, 0) + 1
}

function getSnpPinNames(value: string | undefined): string[] {
  const count = extractPortCountFromSnp(value)
  return generateSnpPinNames(count)
}

function normalizeBlock(block: Block, index: number): Block {
  const type = block.type ?? "snp"
  const meta = BLOCK_META[type]
  const pinNames = type === "snp" ? getSnpPinNames(block.value) : meta.pinNames
  return {
    ...block,
    label: block.label ?? meta.label,
    pins: block.pins?.length === pinNames.length ? block.pins : buildPins(pinNames),
    position: block.position ?? getDefaultPosition(index),
    rotation: block.rotation ?? meta.rotation,
    type,
    value: block.value ?? meta.value,
  }
}

export function createBlock(type: CircuitBlockType, blocks: Block[], position?: Point): Block {
  const index = nextOrdinal(type, blocks)
  const meta = BLOCK_META[type]
  return normalizeBlock(
    {
      id: `${meta.prefix}${index}`,
      label: meta.label,
      position,
      type,
      value: meta.valueFactory ? meta.valueFactory(index) : meta.value,
    },
    blocks.length
  )
}

export function normalizeCircuitData(data: CircuitData): CircuitData {
  return {
    blocks: data.blocks.map(normalizeBlock),
    links: data.links,
  }
}

export function syncPositions(blocks: Block[], previous: Record<string, Point>) {
  return Object.fromEntries(
    blocks.map((block, index) => [block.id, block.position ?? previous[block.id] ?? getDefaultPosition(index)])
  )
}
