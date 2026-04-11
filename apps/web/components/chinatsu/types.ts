export type CircuitBlockType = "capacitor" | "ground" | "inductor" | "port" | "resistor" | "snp"
export type Rotation = 0 | 90 | 180 | 270

export interface Point {
  x: number
  y: number
}

export interface Rect extends Point {
  width: number
  height: number
}

export interface BlockPin {
  name?: string
}

export interface Block {
  id: string
  label?: string
  pins?: BlockPin[]
  position?: Point
  rotation?: Rotation
  type?: CircuitBlockType
  value?: string
}

export interface BlockLinkPin {
  kind?: "pin"
  node: string
  pin: number
}

export interface PointLinkPin extends Point {
  kind: "point"
}

export type LinkPin = BlockLinkPin | PointLinkPin

export interface Link {
  from: LinkPin
  points?: Point[]
  to: LinkPin
}

export interface CircuitData {
  blocks: Block[]
  links: Link[]
}

export interface PaletteItem {
  hint: string
  label: string
  type: CircuitBlockType
}

export interface RoutedLink {
  key: string
  link: Link
  points: Point[]
  routeIndex: number
}

export interface BlockLayout {
  height: number
  pinPoints: Point[]
  width: number
}

export const GRID_SIZE = 20
export const CANVAS_WIDTH = 1640
export const CANVAS_HEIGHT = 920
