/**
 * Antenna Builder Module
 *
 * Extracted from antenna.tsx for standalone use in batch generation.
 * Generates 3D antenna geometries using manifold-3d operations.
 */

import type { Manifold, ManifoldToplevel } from "manifold-3d"

export type EdgePosition = "left" | "right" | "top" | "bottom"

export type AntennaCut = {
    position: EdgePosition
    distance: number
    width: number
}

export type AntennaNib = {
    position: EdgePosition
    distance: number
    thickness: number
    translate: [number, number, number]
    width: number
}

export type AntennaOptions = {
    frame: {
        width: number
        gap: number
        cuts: AntennaCut[]
        nibs: AntennaNib[]
    }
}

export type GeneratedAntenna = {
    antennaFrame: Manifold
    inner: Manifold
}

const EDGE_ORDER: EdgePosition[] = ["top", "bottom", "left", "right"]

function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value))
}

function randomBetween(min: number, max: number): number {
    return min + Math.random() * (max - min)
}

function shuffledEdges(): EdgePosition[] {
    const edges = [...EDGE_ORDER]
    for (let i = edges.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[edges[i], edges[j]] = [edges[j], edges[i]]
    }
    return edges
}

function crossSizeForEdge(position: EdgePosition, baseWidth: number, baseHeight: number): number {
    return position === "left" || position === "right" ? baseHeight : baseWidth
}

function pickOffset(
    crossSize: number,
    spanWidth: number,
    occupied: { center: number; halfWidth: number }[],
    margin: number
): number {
    const limit = Math.max(crossSize / 2 - spanWidth / 2, 0)
    let best = 0
    let bestClearance = Number.NEGATIVE_INFINITY

    for (let attempt = 0; attempt < 32; attempt++) {
        const candidate = randomBetween(-limit, limit)
        const clearance = occupied.length === 0
            ? Number.POSITIVE_INFINITY
            : Math.min(...occupied.map(item => Math.abs(candidate - item.center) - item.halfWidth - spanWidth / 2 - margin))
        if (clearance >= 0) {
            return candidate
        }
        if (clearance > bestClearance) {
            best = candidate
            bestClearance = clearance
        }
    }

    return best
}

function createEdgeSegment(
    module: ManifoldToplevel,
    size: { x: number, y: number, z: number },
    position: EdgePosition,
    distance: number,
    thickness: number,
    length: number,
    depth: number
): Manifold {
    const { Manifold } = module

    if (position === "left" || position === "right") {
        const x = (position === "left" ? -1 : 1) * (size.x / 2 - thickness / 2)
        const y = clamp(distance, -(size.y - length) / 2, (size.y - length) / 2)
        return Manifold.cube([thickness, length, depth], true).translate([x, y, 0])
    }

    const x = clamp(distance, -(size.x - length) / 2, (size.x - length) / 2)
    const y = (position === "bottom" ? -1 : 1) * (size.y / 2 - thickness / 2)
    return Manifold.cube([length, thickness, depth], true).translate([x, y, 0])
}

export type AntennaFeedPlacement = {
    axis: "x" | "y"
    direction: -1 | 1
    crossAxis: number
    bodyEdge: number
    nibEdge: number
    frameEdge: number
    spanCenter: number
    spanLength: number
}

export function getAntennaFeedPlacement(
    innerSize: { x: number, y: number, z: number },
    gap: number,
    frameWidth: number,
    nib: Pick<AntennaNib, "position" | "distance" | "width">
): AntennaFeedPlacement {
    const axis = nib.position === "left" || nib.position === "right" ? "x" : "y"
    const direction = nib.position === "left" || nib.position === "bottom" ? -1 : 1
    const axisSize = axis === "x" ? innerSize.x : innerSize.y
    const crossSize = axis === "x" ? innerSize.y : innerSize.x
    const crossLimit = Math.max((crossSize - nib.width) / 2, 0)
    const bodyEdge = direction * Math.max((axisSize - gap) / 2, 0)
    const frameEdge = direction * (axisSize / 2)
    const availableGap = Math.abs(frameEdge - bodyEdge)
    const portGap = clamp(frameWidth / 2, availableGap * 0.2, availableGap * 0.8)
    const nibEdge = frameEdge - direction * portGap
    const spanCenter = (bodyEdge + nibEdge) / 2
    const spanLength = Math.max(Math.abs(nibEdge - bodyEdge), 0.0001)

    return {
        axis,
        direction,
        crossAxis: clamp(nib.distance, -crossLimit, crossLimit),
        bodyEdge,
        nibEdge,
        frameEdge,
        spanCenter,
        spanLength,
    }
}

/**
 * Create a nib that extends from inner to antennaFrame edge
 * The feed gap stays open so the port can bridge the nib to the frame.
 */
function createNibForFrame(
    module: ManifoldToplevel,
    innerSize: { x: number, y: number, z: number },
    gap: number,
    position: EdgePosition,
    distance: number,
    frameWidth: number,
    nibWidth: number,
    depth: number,
    translate: [number, number, number]
): Manifold {
    const { Manifold } = module
    const placement = getAntennaFeedPlacement(
        innerSize,
        gap,
        frameWidth,
        { position, distance, width: nibWidth }
    )

    if (placement.axis === "x") {
        return Manifold.cube([placement.spanLength, nibWidth, depth], true)
            .translate([placement.spanCenter + translate[0], placement.crossAxis + translate[1], translate[2]])
    }

    return Manifold.cube([nibWidth, placement.spanLength, depth], true)
        .translate([placement.crossAxis + translate[0], placement.spanCenter + translate[1], translate[2]])
}

export function buildAntenna(
    phone: Manifold,
    module: ManifoldToplevel,
    opts: AntennaOptions
): GeneratedAntenna {
    const { Manifold } = module
    const bound = phone.boundingBox()
    const size = {
        x: bound.max[0] - bound.min[0],
        y: bound.max[1] - bound.min[1],
        z: bound.max[2] - bound.min[2],
    }

    const cavity = Manifold.cube([
        Math.max(size.x - opts.frame.width * 2, opts.frame.width),
        Math.max(size.y - opts.frame.width * 2, opts.frame.width),
        size.z + 0.12,
    ], true)

    let antennaFrame = phone.subtract(cavity)

    for (const cut of opts.frame.cuts) {
        antennaFrame = antennaFrame.subtract(
            createEdgeSegment(
                module,
                size,
                cut.position,
                cut.distance,
                opts.frame.width * 1.8,
                cut.width,
                size.z + 0.2
            )
        )
    }

    const innerSize = {
        x: Math.max(size.x - opts.frame.width * 2, opts.frame.width),
        y: Math.max(size.y - opts.frame.width * 2, opts.frame.width),
        z: Math.max(size.z * 0.24, 0.12),
    }

    let inner = Manifold.cube([
        innerSize.x - opts.frame.gap,
        innerSize.y - opts.frame.gap,
        innerSize.z
    ], true).translate([0, 0, -size.z * 0.08])

    for (const nib of opts.frame.nibs) {
        // Keep the nib attached to the inner body while leaving an open feed gap to the frame.
        const nibManifold = createNibForFrame(
            module,
            innerSize,
            opts.frame.gap,
            nib.position,
            nib.distance,
            opts.frame.width,
            nib.width,
            innerSize.z,
            nib.translate
        )
        inner = inner.add(nibManifold)
    }

    return { antennaFrame, inner }
}

export function manifoldToMeshData(manifold: Manifold): {
    verts: number[][]
    faces: number[][]
} {
    const mesh = manifold.getMesh()
    const verts: number[][] = []
    const faces: number[][] = []

    // Extract vertices (3 floats per vertex)
    for (let i = 0; i < mesh.vertProperties.length; i += mesh.numProp) {
        verts.push([
            mesh.vertProperties[i],
            mesh.vertProperties[i + 1],
            mesh.vertProperties[i + 2],
        ])
    }

    // Extract faces (3 indices per triangle)
    for (let i = 0; i < mesh.triVerts.length; i += 3) {
        faces.push([
            mesh.triVerts[i],
            mesh.triVerts[i + 1],
            mesh.triVerts[i + 2],
        ])
    }

    return { verts, faces }
}

export function generateRandomAntennaOptions(
    baseWidth: number,
    baseHeight: number,
    baseDepth: number,
    numNibs: number = 3  // Configurable number of nibs, default 3
): AntennaOptions {
    const frameWidth = baseWidth * (0.04 + Math.random() * 0.03) // 4-7% of width
    const gap = baseWidth * (0.08 + Math.random() * 0.06) // 8-14% of width
    const nibOccupied = {
        top: [] as { center: number; halfWidth: number }[],
        bottom: [] as { center: number; halfWidth: number }[],
        left: [] as { center: number; halfWidth: number }[],
        right: [] as { center: number; halfWidth: number }[],
    }
    const nibs: AntennaNib[] = Array.from({ length: numNibs }, (_, index) => {
        const position = EDGE_ORDER[index % EDGE_ORDER.length]
        const crossSize = crossSizeForEdge(position, baseWidth, baseHeight)
        const width = randomBetween(crossSize * 0.18, crossSize * 0.42)
        const distance = pickOffset(crossSize, width, nibOccupied[position], crossSize * 0.06)
        nibOccupied[position].push({ center: distance, halfWidth: width / 2 })
        return { position, distance, thickness: baseDepth, translate: [0, 0, 0], width }
    })

    const cutCount = 1 + Math.floor(Math.random() * EDGE_ORDER.length)
    const cutSides = shuffledEdges().slice(0, cutCount)
    const cuts: AntennaCut[] = cutSides.map(position => {
        const crossSize = crossSizeForEdge(position, baseWidth, baseHeight)
        const width = randomBetween(crossSize * 0.10, crossSize * 0.24)
        const distance = pickOffset(crossSize, width, nibOccupied[position], crossSize * 0.04)
        nibOccupied[position].push({ center: distance, halfWidth: width / 2 })
        return { position, distance, width }
    })

    return {
        frame: {
            width: frameWidth,
            gap,
            cuts,
            nibs,
        },
    }
}

export function createPhoneBase(
    module: ManifoldToplevel,
    width: number,
    height: number,
    depth: number
): Manifold {
    const { Manifold } = module
    return Manifold.cube([width, height, depth], true)
}
