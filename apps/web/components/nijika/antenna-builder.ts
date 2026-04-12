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

function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value))
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
        inner = inner.add(
            createEdgeSegment(
                module,
                innerSize,
                nib.position,
                nib.distance,
                opts.frame.width,
                nib.width,
                innerSize.z
            ).translate(nib.translate)
        )
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

    // Generate random cuts (2-6 cuts)
    const numCuts = 2 + Math.floor(Math.random() * 5)
    const cuts: AntennaCut[] = []
    const positions: EdgePosition[] = ["top", "bottom", "left", "right"]

    for (let i = 0; i < numCuts; i++) {
        const position = positions[Math.floor(Math.random() * positions.length)]
        const distanceRange = position === "left" || position === "right"
            ? baseHeight * 0.4
            : baseWidth * 0.4

        cuts.push({
            position,
            distance: (Math.random() * 2 - 1) * distanceRange,
            width: 0.1 + Math.random() * 0.25,
        })
    }

    // Generate nibs at configurable positions (distribute across different edges)
    const nibs: AntennaNib[] = []

    for (let i = 0; i < numNibs; i++) {
        // Distribute nibs across edges, allowing multiple per edge if numNibs > 4
        const position = positions[i % positions.length]
        const distanceRange = position === "left" || position === "right"
            ? baseHeight * 0.35
            : baseWidth * 0.35

        nibs.push({
            position,
            distance: (Math.random() * 2 - 1) * distanceRange,
            thickness: baseDepth * (0.8 + Math.random() * 0.4),
            translate: [
                (Math.random() - 0.5) * 0.1,
                (Math.random() - 0.5) * 0.1,
                (Math.random() - 0.5) * 0.1,
            ],
            width: 0.2 + Math.random() * 0.25,
        })
    }

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
