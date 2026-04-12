#!/usr/bin/env tsx
/**
 * Batch Antenna Generator and S-Parameter Extractor
 *
 * Generates N random antenna structures, exports them as JSON for nijika.exe,
 * runs simulations, and collects S-parameters for AI model training.
 */

import * as fs from "fs/promises"
import * as path from "path"
import { execSync } from "child_process"
import { getManifoldModule } from "manifold-3d/lib/wasm.js"
import type { ManifoldToplevel } from "manifold-3d"
import {
    buildAntenna,
    createPhoneBase,
    generateRandomAntennaOptions,
    manifoldToMeshData,
} from "../components/nijika/antenna-builder.js"

const NIJIKA_EXE = path.resolve("../../build/Debug/nijika.exe")
const OUTPUT_DIR = path.resolve("../../tmp/antenna-dataset")
const NUM_SAMPLES = 10
const NUM_NIBS = 3  // Configurable: number of nibs per antenna

// Simulation parameters for nijika.exe
const SIMULATION_CONFIG = {
    freq: {
        min: 0.0,
        max: 5000000000.0, // 5 GHz
    },
    units: {
        geometry: 0.001, // mm
        time: 1e-9,
        frequency: 1000000000.0, // 1 GHz
    },
    fitd: {
        data_type: "fp32" as const,
        loop_steps: 3000,
        timestep: 0.0,
    },
    pml: {
        lower: {
            x: [4, 0.005],
            y: [4, 0.005],
            z: [4, 0.005],
        },
        upper: {
            x: [4, 0.005],
            y: [4, 0.005],
            z: [4, 0.005],
        },
    },
}

// Base phone dimensions (in mm, will be scaled by geometry unit)
const PHONE_DIMS = {
    width: 80,  // mm
    height: 160, // mm
    depth: 8,    // mm
}

function scaleToUnits(mm: number): number {
    return mm * SIMULATION_CONFIG.units.geometry
}

async function ensureDir(dir: string) {
    try {
        await fs.mkdir(dir, { recursive: true })
    } catch {
        // Directory already exists
    }
}

async function runNijikaSimulation(jsonPath: string, outputDir: string): Promise<{
    success: boolean
    sParams?: Record<string, number[]>
    error?: string
}> {
    const baseName = path.basename(jsonPath, ".json")

    try {
        // Run nijika.exe
        const cmd = [
            `"${NIJIKA_EXE}"`,
            "start",
            `"${jsonPath}"`,
            "--verbose",
            "--mesh-hex-build-grid",
            "--mesh-hex-grid-padding-cells", "2",
            "--mesh-hex-cell-per-wavelength", "20",
            "--fitd-loop-steps", SIMULATION_CONFIG.fitd.loop_steps.toString(),
        ].join(" ")

        console.log(`  Running: ${cmd}`)
        execSync(cmd, { stdio: "pipe", cwd: path.dirname(jsonPath) })

        // Parse S-parameters from output (CST format)
        const sParamPath = path.join(outputDir, "S1,1.cst.txt")
        let sParams: Record<string, number[]> = {}

        try {
            const sParamContent = await fs.readFile(sParamPath, "utf-8")
            const lines = sParamContent.split("\n")

            const freqs: number[] = []
            const realParts: number[] = []
            const imagParts: number[] = []

            for (const line of lines) {
                const trimmed = line.trim()
                // Skip header lines (non-numeric start)
                if (!trimmed || isNaN(parseFloat(trimmed.charAt(0)))) {
                    continue
                }
                const parts = trimmed.split(/\s+/)
                if (parts.length >= 3) {
                    freqs.push(parseFloat(parts[0]))
                    realParts.push(parseFloat(parts[1]))
                    imagParts.push(parseFloat(parts[2]))
                }
            }

            sParams = {
                frequency: freqs,
                s11_real: realParts,
                s11_imag: imagParts,
            }
        } catch {
            console.log("  Warning: Could not parse S-parameters")
        }

        return { success: true, sParams }
    } catch (error) {
        return {
            success: false,
            error: error instanceof Error ? error.message : String(error),
        }
    }
}

function generatePorts(
    nibs: { position: "left" | "right" | "top" | "bottom"; distance: number; thickness: number }[],
    phoneDims: { width: number; height: number; depth: number },
    frameWidth: number
): { num: number; label: string; impedance: number; positions: { from: { x: number; y: number; z: number }; to: { x: number; y: number; z: number } }[] }[] {
    return nibs.map((nib, index) => {
        const halfW = scaleToUnits(phoneDims.width / 2)
        const halfH = scaleToUnits(phoneDims.height / 2)
        const halfD = scaleToUnits(phoneDims.depth / 2)
        const gap = frameWidth * 0.5  // Port sits between nib and frame
        const nibZ = halfD * 0.1  // Slightly inside from surface

        let from: { x: number; y: number; z: number }
        let to: { x: number; y: number; z: number }

        switch (nib.position) {
            case "left":
                from = { x: -halfW, y: nib.distance, z: nibZ }
                to = { x: -halfW + gap, y: nib.distance, z: nibZ }
                break
            case "right":
                from = { x: halfW, y: nib.distance, z: nibZ }
                to = { x: halfW - gap, y: nib.distance, z: nibZ }
                break
            case "top":
                from = { x: nib.distance, y: halfH, z: nibZ }
                to = { x: nib.distance, y: halfH - gap, z: nibZ }
                break
            case "bottom":
                from = { x: nib.distance, y: -halfH, z: nibZ }
                to = { x: nib.distance, y: -halfH + gap, z: nibZ }
                break
        }

        return {
            num: index + 1,
            label: `port${index + 1}`,
            impedance: 50.0,
            positions: [{ from, to }],
        }
    })
}

async function generateAntennaSample(
    module: ManifoldToplevel,
    index: number
): Promise<{
    config: Record<string, unknown>
    geometry: {
        verts: number[][]
        faces: number[][]
    }
}> {
    // Create base phone shape
    const phoneBase = createPhoneBase(
        module,
        scaleToUnits(PHONE_DIMS.width),
        scaleToUnits(PHONE_DIMS.height),
        scaleToUnits(PHONE_DIMS.depth)
    )

    // Generate random antenna options with fixed number of nibs
    const options = generateRandomAntennaOptions(
        scaleToUnits(PHONE_DIMS.width),
        scaleToUnits(PHONE_DIMS.height),
        scaleToUnits(PHONE_DIMS.depth),
        NUM_NIBS
    )

    // Build antenna geometry
    const { antennaFrame } = buildAntenna(phoneBase, module, options)
    const meshData = manifoldToMeshData(antennaFrame)

    // Create JSON config for nijika.exe
    const config = {
        ...SIMULATION_CONFIG,
        hex: {
            grid: {
                x: [-0.01, 0.0, 0.01],
                y: [-0.01, 0.0, 0.01],
                z: [-0.02, 0.0, 0.02],
            },
        },
        materials: [
            {
                name: "Vacuum",
                type: "Normal",
                eps: { x: 1.0, y: 1.0, z: 1.0 },
                tand: { value: 0.0, freq: 0.0 },
            },
            {
                name: "PEC",
                type: "PEC",
                eps: { x: 1.0, y: 1.0, z: 1.0 },
                tand: { value: 0.0, freq: 0.0 },
            },
        ],
        solids: [
            {
                name: "antenna:frame",
                material: "PEC",
            },
        ],
        mesh: meshData,
        ports: generatePorts(options.frame.nibs, PHONE_DIMS, options.frame.width),
        // Store antenna configuration for ML training
        antennaConfig: {
            frameWidth: options.frame.width,
            gap: options.frame.gap,
            numCuts: options.frame.cuts.length,
            numNibs: options.frame.nibs.length,
            cuts: options.frame.cuts,
            nibs: options.frame.nibs,
        },
    }

    // Clean up
    phoneBase.delete()
    antennaFrame.delete()

    return { config, geometry: meshData }
}

async function main() {
    console.log("=" .repeat(60))
    console.log("Batch Antenna Generator and S-Parameter Extractor")
    console.log("=" .repeat(60))
    console.log()

    // Check nijika.exe exists
    try {
        await fs.access(NIJIKA_EXE)
    } catch {
        console.error(`Error: nijika.exe not found at ${NIJIKA_EXE}`)
        process.exit(1)
    }

    // Initialize output directory
    await ensureDir(OUTPUT_DIR)
    console.log(`Output directory: ${OUTPUT_DIR}`)
    console.log(`Generating ${NUM_SAMPLES} random antenna samples...`)
    console.log()

    // Load manifold module
    console.log("Loading manifold-3d module...")
    const module = await getManifoldModule()
    console.log("Module loaded.")
    console.log()

    // Generate samples
    const results: {
        index: number
        config: Record<string, unknown>
        geometry: { verts: number[][]; faces: number[][] }
        simulation?: {
            success: boolean
            sParams?: Record<string, number[]>
            error?: string
        }
    }[] = []

    for (let i = 0; i < NUM_SAMPLES; i++) {
        console.log(`[${i + 1}/${NUM_SAMPLES}] Generating antenna...`)

        // Generate antenna
        const { config, geometry } = await generateAntennaSample(module, i)

        // Save JSON
        const jsonName = `antenna_${String(i).padStart(3, "0")}.json`
        const jsonPath = path.join(OUTPUT_DIR, jsonName)
        await fs.writeFile(jsonPath, JSON.stringify(config, null, 2))
        console.log(`  Saved: ${jsonName}`)

        // Run simulation
        const outputDir = path.join(OUTPUT_DIR, `antenna_${String(i).padStart(3, "0")}`)
        console.log(`  Running simulation...`)
        const simulation = await runNijikaSimulation(jsonPath, outputDir)

        if (simulation.success) {
            console.log(`  Simulation completed successfully`)
        } else {
            console.log(`  Simulation failed: ${simulation.error}`)
        }

        results.push({ index: i, config, geometry, simulation })
        console.log()
    }

    // Generate summary
    console.log("=" .repeat(60))
    console.log("Summary")
    console.log("=" .repeat(60))
    console.log()

    const successful = results.filter(r => r.simulation?.success).length
    console.log(`Total samples: ${NUM_SAMPLES}`)
    console.log(`Successful simulations: ${successful}`)
    console.log(`Failed simulations: ${NUM_SAMPLES - successful}`)
    console.log()

    // Save dataset metadata
    const dataset = results.map(r => ({
        index: r.index,
        config: r.config.antennaConfig,
        sParams: r.simulation?.sParams,
        success: r.simulation?.success,
        jsonFile: `antenna_${String(r.index).padStart(3, "0")}.json`,
    }))

    const datasetPath = path.join(OUTPUT_DIR, "dataset.json")
    await fs.writeFile(datasetPath, JSON.stringify(dataset, null, 2))
    console.log(`Dataset metadata saved to: ${datasetPath}`)

    // Cleanup manifold module
    // Note: manifold-3d doesn't have explicit cleanup, relies on GC
}

main().catch(error => {
    console.error("Fatal error:", error)
    process.exit(1)
})
