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
    type AntennaNib,
    buildAntenna,
    createPhoneBase,
    generateRandomAntennaOptions,
    getAntennaFeedPlacement,
    manifoldToMeshData,
} from "../components/nijika/antenna-builder.js"

const NIJIKA_EXE = path.resolve("../../build/Debug/nijika.exe")
const DEFAULT_OUTPUT_DIR = path.resolve("../../tmp/antenna-dataset")
const DEFAULT_NUM_SAMPLES = 100
const DEFAULT_NUM_NIBS = 3
const ESSENTIAL_OUTPUT_RE = /^S\d+,\d+\.cst\.txt$/

type CliOptions = {
    append: boolean
    keepIntermediates: boolean
    numNibs: number
    outputDir: string
    samples: number
}

type DatasetEntry = {
    config: unknown
    index: number
    jsonFile: string
    sParams?: Record<string, number[]>
    success: boolean
}

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
        loop_steps: 10000,
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

function parseCliOptions(argv: string[]): CliOptions {
    const options: CliOptions = {
        append: false,
        keepIntermediates: false,
        numNibs: DEFAULT_NUM_NIBS,
        outputDir: DEFAULT_OUTPUT_DIR,
        samples: DEFAULT_NUM_SAMPLES,
    }

    for (let index = 0; index < argv.length; index++) {
        const arg = argv[index]
        const next = argv[index + 1]
        const readValue = () => {
            if (!next || next.startsWith("--")) {
                throw new Error(`Missing value for ${arg}`)
            }
            index += 1
            return next
        }

        if (arg === "--append") {
            options.append = true
            continue
        }
        if (arg === "--keep-intermediates") {
            options.keepIntermediates = true
            continue
        }
        if (arg === "--samples") {
            options.samples = Number.parseInt(readValue(), 10)
            continue
        }
        if (arg.startsWith("--samples=")) {
            options.samples = Number.parseInt(arg.slice("--samples=".length), 10)
            continue
        }
        if (arg === "--nibs") {
            options.numNibs = Number.parseInt(readValue(), 10)
            continue
        }
        if (arg.startsWith("--nibs=")) {
            options.numNibs = Number.parseInt(arg.slice("--nibs=".length), 10)
            continue
        }
        if (arg === "--output-dir") {
            options.outputDir = path.resolve(readValue())
            continue
        }
        if (arg.startsWith("--output-dir=")) {
            options.outputDir = path.resolve(arg.slice("--output-dir=".length))
            continue
        }
        if (arg === "--help") {
            console.log("Usage: tsx scripts/batch-antenna.ts [--samples N] [--nibs N] [--append] [--keep-intermediates] [--output-dir DIR]")
            process.exit(0)
        }
        throw new Error(`Unknown argument: ${arg}`)
    }

    if (!Number.isInteger(options.samples) || options.samples <= 0) {
        throw new Error(`Invalid --samples value: ${options.samples}`)
    }
    if (!Number.isInteger(options.numNibs) || options.numNibs <= 0) {
        throw new Error(`Invalid --nibs value: ${options.numNibs}`)
    }
    return options
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

async function clearDatasetOutputs(root: string) {
    const entries = await fs.readdir(root, { withFileTypes: true })
    await Promise.all(entries.map(async entry => {
        const shouldDelete = entry.name === "dataset.json"
            || /^antenna_\d+\.json$/.test(entry.name)
            || (entry.isDirectory() && /^antenna_\d+$/.test(entry.name))
        if (!shouldDelete) {
            return
        }
        await fs.rm(path.join(root, entry.name), { recursive: true, force: true })
    }))
}

async function pruneSimulationOutputs(outputDir: string) {
    const entries = await fs.readdir(outputDir, { withFileTypes: true })
    await Promise.all(entries.map(async entry => {
        if (!entry.isFile() || ESSENTIAL_OUTPUT_RE.test(entry.name)) {
            return
        }
        await fs.rm(path.join(outputDir, entry.name), { force: true })
    }))
    await fs.rm(path.join(outputDir, "plrc"), { recursive: true, force: true })
}

async function inferNextSampleIndex(root: string) {
    const entries = await fs.readdir(root, { withFileTypes: true })
    let maxIndex = -1
    for (const entry of entries) {
        const match = entry.name.match(/^antenna_(\d+)(?:\.json)?$/)
        if (!match) {
            continue
        }
        maxIndex = Math.max(maxIndex, Number.parseInt(match[1], 10))
    }
    return maxIndex + 1
}

function formatDuration(ms: number) {
    const totalSeconds = Math.max(0, Math.round(ms / 1000))
    const hours = Math.floor(totalSeconds / 3600)
    const minutes = Math.floor((totalSeconds % 3600) / 60)
    const seconds = totalSeconds % 60
    return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`
}

async function runNijikaSimulation(jsonPath: string, outputDir: string): Promise<{
    success: boolean
    sParams?: Record<string, number[]>
    error?: string
}> {
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
        const sParams = await readSParamSummary(outputDir)
        if (!sParams) {
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

async function readSParamSummary(outputDir: string): Promise<Record<string, number[]> | undefined> {
    const sParamPath = path.join(outputDir, "S1,1.cst.txt")
    try {
        const sParamContent = await fs.readFile(sParamPath, "utf-8")
        const lines = sParamContent.split("\n")
        const freqs: number[] = []
        const realParts: number[] = []
        const imagParts: number[] = []

        for (const line of lines) {
            const trimmed = line.trim()
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

        return {
            frequency: freqs,
            s11_real: realParts,
            s11_imag: imagParts,
        }
    } catch {
        return undefined
    }
}

async function readSampleMetadata(root: string, index: number): Promise<DatasetEntry> {
    const sampleName = `antenna_${String(index).padStart(3, "0")}`
    const jsonFile = `${sampleName}.json`
    const config = JSON.parse(await fs.readFile(path.join(root, jsonFile), "utf-8"))
    const portCount = Array.isArray(config.ports) ? config.ports.length : 0
    const sampleDir = path.join(root, sampleName)
    const needed = Array.from({ length: portCount }, (_, row) =>
        Array.from({ length: portCount }, (_, col) => path.join(sampleDir, `S${row + 1},${col + 1}.cst.txt`))
    ).flat()
    const success = needed.length > 0 && await Promise.all(needed.map(file => fs.access(file).then(() => true).catch(() => false)))
        .then(results => results.every(Boolean))
    const metadata: DatasetEntry = {
        config: config.antennaConfig,
        index,
        jsonFile,
        success,
    }
    if (success) {
        metadata.sParams = await readSParamSummary(sampleDir)
    }
    return metadata
}

async function rebuildDatasetMetadata(root: string): Promise<DatasetEntry[]> {
    const entries = await fs.readdir(root, { withFileTypes: true })
    const indices = entries
        .filter(entry => entry.isFile() && /^antenna_\d+\.json$/.test(entry.name))
        .map(entry => Number.parseInt(entry.name.slice("antenna_".length, -".json".length), 10))
        .sort((left, right) => left - right)
    const dataset: DatasetEntry[] = []
    for (const index of indices) {
        dataset.push(await readSampleMetadata(root, index))
    }
    return dataset
}

function generatePorts(
    nibs: AntennaNib[],
    phoneDims: { width: number; height: number; depth: number },
    frameWidth: number,
    gap: number
): { num: number; label: string; impedance: number; positions: { from: { x: number; y: number; z: number }; to: { x: number; y: number; z: number } }[] }[] {
    const innerSize = {
        x: scaleToUnits(phoneDims.width) - frameWidth * 2,
        y: scaleToUnits(phoneDims.height) - frameWidth * 2,
        z: Math.max(scaleToUnits(phoneDims.depth) * 0.24, 0.12),
    }
    const meshZ = scaleToUnits(phoneDims.depth) / 2

    return nibs.map((nib, index) => {
        const placement = getAntennaFeedPlacement(innerSize, gap, frameWidth, nib)
        const portGap = Math.abs(placement.frameEdge - placement.nibEdge)
        const conductorInset = Math.min(frameWidth * 0.15, portGap * 0.25)
        let from: { x: number; y: number; z: number }
        let to: { x: number; y: number; z: number }

        if (placement.axis === "x") {
            from = {
                x: placement.nibEdge - placement.direction * conductorInset,
                y: placement.crossAxis,
                z: meshZ,
            }
            to = {
                x: placement.frameEdge + placement.direction * conductorInset,
                y: placement.crossAxis,
                z: meshZ,
            }
        } else {
            from = {
                x: placement.crossAxis,
                y: placement.nibEdge - placement.direction * conductorInset,
                z: meshZ,
            }
            to = {
                x: placement.crossAxis,
                y: placement.frameEdge + placement.direction * conductorInset,
                z: meshZ,
            }
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
    numNibs: number
): Promise<{
    config: Record<string, unknown>
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
        numNibs
    )

    // Build antenna geometry - combine frame and inner (nibs) into single manifold
    const { antennaFrame, inner } = buildAntenna(phoneBase, module, options)
    const combinedGeometry = antennaFrame.add(inner)
    const meshData = manifoldToMeshData(combinedGeometry)

    // Create JSON config for nijika.exe
    const config = {
        ...SIMULATION_CONFIG,
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
                name: "antenna:combined",
                material: "PEC",
            },
        ],
        mesh: meshData,
        ports: generatePorts(options.frame.nibs, PHONE_DIMS, options.frame.width, options.frame.gap),
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
    inner.delete()
    combinedGeometry.delete()

    return { config }
}

async function main() {
    const options = parseCliOptions(process.argv.slice(2))
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
    await ensureDir(options.outputDir)
    if (!options.append) {
        await clearDatasetOutputs(options.outputDir)
    }
    const startIndex = options.append ? await inferNextSampleIndex(options.outputDir) : 0
    console.log(`Output directory: ${options.outputDir}`)
    console.log(`Generating ${options.samples} random antenna samples...`)
    console.log(`Starting index: ${startIndex}`)
    console.log(`Nib count per antenna: ${options.numNibs}`)
    console.log(`Keep intermediates: ${options.keepIntermediates ? "yes" : "no"}`)
    console.log(`Append mode: ${options.append ? "yes" : "no"}`)
    console.log()

    // Load manifold module
    console.log("Loading manifold-3d module...")
    const module = await getManifoldModule()
    console.log("Module loaded.")
    console.log()

    // Generate samples
    let successful = 0
    const startedAt = Date.now()

    for (let i = 0; i < options.samples; i++) {
        const sampleIndex = startIndex + i
        console.log(`[${i + 1}/${options.samples}] Generating antenna ${sampleIndex}...`)

        // Generate antenna
        const { config } = await generateAntennaSample(module, options.numNibs)

        // Save JSON
        const jsonName = `antenna_${String(sampleIndex).padStart(3, "0")}.json`
        const jsonPath = path.join(options.outputDir, jsonName)
        await fs.writeFile(jsonPath, JSON.stringify(config, null, 2))
        console.log(`  Saved: ${jsonName}`)

        // Run simulation
        const outputDir = path.join(options.outputDir, `antenna_${String(sampleIndex).padStart(3, "0")}`)
        console.log(`  Running simulation...`)
        const simulation = await runNijikaSimulation(jsonPath, outputDir)

        if (simulation.success) {
            successful += 1
            if (!options.keepIntermediates) {
                await pruneSimulationOutputs(outputDir)
            }
            console.log(`  Simulation completed successfully`)
        } else {
            console.log(`  Simulation failed: ${simulation.error}`)
        }
        const elapsedMs = Date.now() - startedAt
        const averageMs = elapsedMs / (i + 1)
        const etaMs = averageMs * (options.samples - i - 1)
        console.log(`  Elapsed: ${formatDuration(elapsedMs)} | ETA: ${formatDuration(etaMs)}`)
        console.log()
    }

    // Generate summary
    console.log("=" .repeat(60))
    console.log("Summary")
    console.log("=" .repeat(60))
    console.log()

    console.log(`New samples requested: ${options.samples}`)
    console.log(`Successful simulations: ${successful}`)
    console.log(`Failed simulations: ${options.samples - successful}`)
    console.log()

    // Save dataset metadata
    const dataset = await rebuildDatasetMetadata(options.outputDir)
    const datasetPath = path.join(options.outputDir, "dataset.json")
    await fs.writeFile(datasetPath, JSON.stringify(dataset, null, 2))
    console.log(`Dataset metadata saved to: ${datasetPath}`)

    // Cleanup manifold module
    // Note: manifold-3d doesn't have explicit cleanup, relies on GC
}

main().catch(error => {
    console.error("Fatal error:", error)
    process.exit(1)
})
