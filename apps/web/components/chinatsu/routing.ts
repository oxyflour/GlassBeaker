import { isPointLinkPin, pointKey, simplifyPoints } from "./geometry"
import { getBlockRect, getPinPoint } from "./symbol-geometry"
import { type Block, type CircuitData, type Link, type LinkPin, type Point, type RoutedLink, CANVAS_HEIGHT, CANVAS_WIDTH, GRID_SIZE } from "./types"

type GridPoint = { x: number; y: number }
type StepState = GridPoint & { dir: number; score: number }

const DIRECTIONS: GridPoint[] = [{ x: 1, y: 0 }, { x: -1, y: 0 }, { x: 0, y: 1 }, { x: 0, y: -1 }]

function gridKey(point: GridPoint, dir = -1) {
  return `${point.x}:${point.y}:${dir}`
}

function toGrid(point: Point): GridPoint {
  const x = Math.round(point.x / GRID_SIZE)
  const y = Math.round(point.y / GRID_SIZE)
  // 防御性检查：确保坐标是有效数字
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    console.warn("toGrid: invalid coordinates", point)
    return { x: 0, y: 0 }
  }
  return { x, y }
}

function fromGrid(point: GridPoint): Point {
  return { x: point.x * GRID_SIZE, y: point.y * GRID_SIZE }
}

function samePoint(left: Point, right: Point) {
  return left.x === right.x && left.y === right.y
}

function addBlockedRect(blocked: Set<string>, block: Block, position: Point, exempt: Set<string>) {
  // 防御性检查
  if (!Number.isFinite(position.x) || !Number.isFinite(position.y)) {
    console.warn("addBlockedRect: invalid position", { block, position })
    return
  }

  const rect = getBlockRect(block, position)
  const left = Math.max(0, Math.ceil((rect.x + 1) / GRID_SIZE))
  const right = Math.min(Math.floor(CANVAS_WIDTH / GRID_SIZE), Math.floor((rect.x + rect.width - 1) / GRID_SIZE))
  const top = Math.max(0, Math.ceil((rect.y + 1) / GRID_SIZE))
  const bottom = Math.min(Math.floor(CANVAS_HEIGHT / GRID_SIZE), Math.floor((rect.y + rect.height - 1) / GRID_SIZE))

  for (let x = left; x <= right; x += 1) {
    for (let y = top; y <= bottom; y += 1) {
      const key = pointKey(fromGrid({ x, y }))
      if (!exempt.has(key)) {
        blocked.add(key)
      }
    }
  }
}

function markRoute(blocked: Set<string>, points: Point[], exempt: Set<string>) {
  for (let index = 1; index < points.length; index += 1) {
    const start = toGrid(points[index - 1])
    const end = toGrid(points[index])
    const stepX = Math.sign(end.x - start.x)
    const stepY = Math.sign(end.y - start.y)
    // Skip if start and end are the same point
    if (stepX === 0 && stepY === 0) {
      continue
    }
    let cursor = { ...start }
    let iterations = 0
    const maxIterations = Math.abs(end.x - start.x) + Math.abs(end.y - start.y) + 10
    while (cursor.x !== end.x || cursor.y !== end.y) {
      if (iterations++ > maxIterations) {
        console.warn("markRoute: exceeded max iterations", { start, end, stepX, stepY })
        break
      }
      const key = pointKey(fromGrid(cursor))
      if (!exempt.has(key)) {
        blocked.add(key)
      }
      cursor = { x: cursor.x + stepX, y: cursor.y + stepY }
    }
  }
}

function reconstruct(leaf: StepState, parents: Map<string, string>) {
  const path: Point[] = []
  let cursor = gridKey(leaf, leaf.dir)
  while (cursor) {
    const [x, y, dir] = cursor.split(":").map(Number)
    path.push(fromGrid({ x, y }))
    cursor = parents.get(gridKey({ x, y }, dir)) ?? ""
  }
  return simplifyPoints(path.reverse())
}

function searchRoute(start: Point, end: Point, blocked: Set<string>) {
  const origin = toGrid(start)
  const target = toGrid(end)

  // 起点等于终点，直接返回
  if (origin.x === target.x && origin.y === target.y) {
    return [start, end]
  }

  const maxX = Math.floor(CANVAS_WIDTH / GRID_SIZE)
  const maxY = Math.floor(CANVAS_HEIGHT / GRID_SIZE)

  // 确保起点和终点在有效范围内
  if (
    origin.x < 0 || origin.x > maxX || origin.y < 0 || origin.y > maxY ||
    target.x < 0 || target.x > maxX || target.y < 0 || target.y > maxY
  ) {
    return null
  }

  const open: StepState[] = [{ ...origin, dir: -1, score: 0 }]
  const parents = new Map<string, string>()
  const best = new Map<string, number>([[gridKey(origin, -1), 0]])
  const visited = new Set<string>()
  let iterations = 0
  const MAX_ITERATIONS = maxX * maxY * 4 // 最大合理迭代次数

  while (open.length > 0) {
    iterations++
    if (iterations > MAX_ITERATIONS) {
      console.warn("A* search exceeded max iterations")
      return null
    }

    open.sort((left, right) => left.score - right.score)
    const current = open.shift()
    if (!current) {
      break
    }

    const currentKey = gridKey(current, current.dir)
    if (visited.has(currentKey)) {
      continue
    }
    visited.add(currentKey)

    if (current.x === target.x && current.y === target.y) {
      return reconstruct(current, parents)
    }

    DIRECTIONS.forEach((direction, dir) => {
      const next = { x: current.x + direction.x, y: current.y + direction.y }
      if (next.x < 0 || next.y < 0 || next.x > maxX || next.y > maxY) {
        return
      }

      const point = fromGrid(next)
      if (blocked.has(pointKey(point)) && (next.x !== target.x || next.y !== target.y)) {
        return
      }

      const stepCost = (best.get(currentKey) ?? 0) + 1 + (current.dir >= 0 && current.dir !== dir ? 0.35 : 0)
      const candidateKey = gridKey(next, dir)
      if (stepCost >= (best.get(candidateKey) ?? Number.POSITIVE_INFINITY)) {
        return
      }

      best.set(candidateKey, stepCost)
      parents.set(candidateKey, currentKey)
      open.push({
        ...next,
        dir,
        score: stepCost + Math.abs(target.x - next.x) + Math.abs(target.y - next.y),
      })
    })
  }

  return null
}

function getRoutePoints(start: Point, end: Point, blocks: Block[], positions: Record<string, Point>, occupied: RoutedLink[]) {
  const exempt = new Set([pointKey(start), pointKey(end)])
  const hardBlocked = new Set<string>()
  blocks.forEach((block) => {
    const position = positions[block.id]
    if (position) {
      addBlockedRect(hardBlocked, block, position, exempt)
    }
  })

  const withRoutes = new Set(hardBlocked)
  occupied.forEach((route) => markRoute(withRoutes, route.points, exempt))
  const gridPath = searchRoute(start, end, withRoutes) ?? searchRoute(start, end, hardBlocked) ?? simplifyPoints([start, end])
  return stitchExactEndpoints(start, end, gridPath)
}

function stitchExactEndpoints(start: Point, end: Point, path: Point[]) {
  if (path.length === 0) {
    return simplifyPoints([start, end])
  }

  const next: Point[] = []
  appendConnector(next, start, path[0])
  path.forEach((point) => {
    if (!samePoint(next[next.length - 1] ?? point, point)) {
      next.push(point)
    }
  })
  appendConnector(next, path[path.length - 1], end)
  return simplifyPoints(next)
}

function appendConnector(points: Point[], from: Point, to: Point) {
  if (points.length === 0) {
    points.push(from)
  }
  if (samePoint(from, to)) {
    return
  }
  if (from.x !== to.x && from.y !== to.y) {
    points.push({ x: to.x, y: from.y })
  }
  points.push(to)
}

function materializeStoredPoints(link: Link, start: Point, end: Point) {
  const path = [start, ...(link.points?.slice(1, -1) ?? []), end]
  const orthogonal: Point[] = [path[0]]
  for (let index = 1; index < path.length; index += 1) {
    const previous = orthogonal[orthogonal.length - 1]
    const next = path[index]
    if (previous.x !== next.x && previous.y !== next.y) {
      orthogonal.push({ x: next.x, y: previous.y })
    }
    orthogonal.push(next)
  }
  return simplifyPoints(orthogonal)
}

export function resolveLinkPinPoint(blocks: Record<string, Block>, positions: Record<string, Point>, pin: LinkPin) {
  if (isPointLinkPin(pin)) {
    // 防御性检查
    if (!Number.isFinite(pin.x) || !Number.isFinite(pin.y)) {
      console.warn("resolveLinkPinPoint: invalid point pin", pin)
      return null
    }
    return { x: pin.x, y: pin.y }
  }
  const block = blocks[pin.node]
  const position = positions[pin.node]
  if (!block || !position) {
    return null
  }
  const result = getPinPoint(block, position, pin.pin)
  // 防御性检查
  if (!Number.isFinite(result.x) || !Number.isFinite(result.y)) {
    console.warn("resolveLinkPinPoint: invalid result", { block, position, pin, result })
    return null
  }
  return result
}

export function buildRoutes(data: CircuitData, positions: Record<string, Point>) {
  const blocks = Object.fromEntries(data.blocks.map((block) => [block.id, block]))
  const routes: RoutedLink[] = []
  data.links.forEach((link, routeIndex) => {
    const start = resolveLinkPinPoint(blocks, positions, link.from)
    const end = resolveLinkPinPoint(blocks, positions, link.to)
    if (!start || !end) {
      return
    }

    routes.push({
      key: `${pointKey(start)}->${pointKey(end)}:${routeIndex}`,
      link,
      points: link.points?.length ? materializeStoredPoints(link, start, end) : getRoutePoints(start, end, data.blocks, positions, routes),
      routeIndex,
    })
  })
  return routes
}

export function buildPreviewRoute(source: LinkPin, cursorPoint: Point, blocks: Block[], positions: Record<string, Point>, occupied: RoutedLink[]) {
  const map = Object.fromEntries(blocks.map((block) => [block.id, block]))
  const start = resolveLinkPinPoint(map, positions, source)
  if (!start) {
    return null
  }
  return getRoutePoints(start, cursorPoint, blocks, positions, occupied)
}

function shiftPoint(point: Point, delta: Point): Point {
  return { x: point.x + delta.x, y: point.y + delta.y }
}

function isLinkConnectedToBlocks(link: Link, blockIds: Set<string>): boolean {
  return (!isPointLinkPin(link.from) && blockIds.has(link.from.node)) ||
         (!isPointLinkPin(link.to) && blockIds.has(link.to.node))
}

/**
 * 简化的路由计算，用于拖拽时快速更新。
 * 只更新与被拖拽器件相连的走线，其他走线直接使用缓存。
 */
export function buildRoutesSimplified(
  data: CircuitData,
  positions: Record<string, Point>,
  dragBlockIds: string[],
  dragDelta: Point,
  cachedRoutes: RoutedLink[]
): RoutedLink[] {
  if (cachedRoutes.length === 0) {
    return []
  }

  const blocks = Object.fromEntries(data.blocks.map((block) => [block.id, block]))
  const dragBlockIdSet = new Set(dragBlockIds)

  return data.links.map((link, routeIndex) => {
    const cached = cachedRoutes[routeIndex]
    // 如果走线不连接被拖拽的器件，直接使用缓存
    if (!isLinkConnectedToBlocks(link, dragBlockIdSet)) {
      return cached
    }

    const start = resolveLinkPinPoint(blocks, positions, link.from)
    const end = resolveLinkPinPoint(blocks, positions, link.to)
    if (!start || !end) {
      return cached
    }

    const isStartOnDragBlock = !isPointLinkPin(link.from) && dragBlockIdSet.has(link.from.node)
    const isEndOnDragBlock = !isPointLinkPin(link.to) && dragBlockIdSet.has(link.to.node)

    let points: Point[]
    const basePoints = cached?.points ?? [start, end]

    if (isStartOnDragBlock && isEndOnDragBlock) {
      // 两端都在拖拽块上，整体平移
      points = basePoints.map((p) => shiftPoint(p, dragDelta))
    } else if (isStartOnDragBlock || isEndOnDragBlock) {
      // 只有一端在拖拽块上：平移中间点
      points = [start, ...basePoints.slice(1, -1).map((p) => shiftPoint(p, dragDelta)), end]
    } else {
      points = basePoints
    }

    return {
      key: `${pointKey(start)}->${pointKey(end)}:${routeIndex}`,
      link,
      points,
      routeIndex,
    }
  })
}
