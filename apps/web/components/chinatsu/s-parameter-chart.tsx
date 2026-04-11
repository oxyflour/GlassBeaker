"use client"

import dynamic from "next/dynamic"
import { useMemo, useState } from "react"
import type { Data, Layout, Config } from "plotly.js"

// Dynamic import to avoid SSR issues with Plotly
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false })

export interface SParameterData {
  frequency: number[] // Hz
  s_matrix: number[][][] // [output_port][input_port][frequency]
}

interface SParameterChartProps {
  data: SParameterData | null
  title?: string
  height?: number
  /**
   * If true, the chart will fill the parent container height.
   * When used with height=0, the component fills available space.
   */
  fillContainer?: boolean
}

type ScaleMode = "linear" | "dB"

function toDB(linear: number): number {
  return 20 * Math.log10(linear)
}

// Predefined colors for S-parameters
const COLORS = [
  "#1f77b4", // blue
  "#ff7f0e", // orange
  "#2ca02c", // green
  "#d62728", // red
  "#9467bd", // purple
  "#8c564b", // brown
  "#e377c2", // pink
  "#7f7f7f", // gray
  "#bcbd22", // olive
  "#17becf", // cyan
  "#ffbb78",
  "#98df8a",
  "#ff9896",
  "#c5b0d5",
  "#c49c94",
  "#f7b6d2",
]

export default function SParameterChart({
  data,
  title = "S-Parameters",
  height = 400,
  fillContainer = false,
}: SParameterChartProps) {
  // When fillContainer is true or height is 0/undefined, use 100% height
  const containerHeight = fillContainer || !height ? "100%" : height
  const [scaleMode, setScaleMode] = useState<ScaleMode>("dB")

  // Get number of ports from data
  const numPorts = data?.s_matrix?.length ?? 0

  // Generate all possible S-parameters for the given port count
  const allSParams = useMemo(() => {
    const params: { key: string; out: number; in: number; label: string }[] = []
    for (let out = 0; out < numPorts; out++) {
      for (let inp = 0; inp < numPorts; inp++) {
        params.push({
          key: `s${out + 1}${inp + 1}`,
          out,
          in: inp,
          label: `S${out + 1}${inp + 1}`,
        })
      }
    }
    return params
  }, [numPorts])

  // Initialize selection state - default only S11 selected
  const [selectedSParams, setSelectedSParams] = useState<Set<string>>(new Set(["s11"]))

  // Update selection when port count changes - keep S11 selected if available
  useMemo(() => {
    if (numPorts > 0) {
      setSelectedSParams((prev) => {
        // Keep existing valid selections, or default to S11
        const valid = new Set<string>()
        for (const key of prev) {
          if (allSParams.some((p) => p.key === key)) {
            valid.add(key)
          }
        }
        if (valid.size === 0 && allSParams.length > 0) {
          valid.add("s11")
        }
        return valid
      })
    }
  }, [numPorts, allSParams])

  const toggleSParam = (key: string) => {
    setSelectedSParams((prev) => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }

  const plotData = useMemo((): Data[] => {
    if (!data || !data.frequency?.length || !data.s_matrix?.length) return []

    const traces: Data[] = []

    const transform = (val: number) =>
      scaleMode === "dB" ? toDB(val) : val

    for (let i = 0; i < allSParams.length; i++) {
      const { key, out, in: inp, label } = allSParams[i]
      if (!selectedSParams.has(key)) continue

      const sValues = data.s_matrix[out]?.[inp]
      if (!sValues || !Array.isArray(sValues) || sValues.length === 0) continue

      traces.push({
        x: data.frequency.map((f) => f / 1e9), // Convert to GHz for display
        y: sValues.map(transform),
        type: "scatter",
        mode: "lines",
        name: label,
        line: { color: COLORS[i % COLORS.length], width: 2 },
      })
    }

    return traces
  }, [data, scaleMode, selectedSParams, allSParams])

  const layout = useMemo((): Partial<Layout> => {
    const yTitle = scaleMode === "dB" ? "Magnitude (dB)" : "Magnitude (linear)"

    return {
      title: {
        text: title,
        font: { color: "#d9e8f4", size: 16 },
      } as Partial<Layout['title']>,
      xaxis: {
        gridcolor: "#1b2433",
        zerolinecolor: "#1b2433",
        type: "linear",
      },
      yaxis: {
        gridcolor: "#1b2433",
        zerolinecolor: "#1b2433",
      },
      paper_bgcolor: "transparent",
      plot_bgcolor: "#0d1420",
      legend: {
        x: 0,
        y: 1,
        bgcolor: "rgba(13, 20, 32, 0.8)",
        bordercolor: "#1b2433",
        borderwidth: 1,
        font: { color: "#d9e8f4" },
      },
      margin: { t: 40, r: 40, b: 60, l: 60 },
      autosize: true,
    }
  }, [title, scaleMode])

  const config = useMemo<Partial<Config>>(
    () => ({
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    }),
    []
  )

  const hasData = data && data.frequency.length > 0

  return (
    <div
      className="w-full h-full overflow-hidden"
      style={{
        background: "linear-gradient(180deg, #102033, #08131f)",
        border: "1px solid rgba(150,180,210,0.18)",
        padding: 16,
      }}
    >
      {/* Controls */}
      <div
        style={{
          display: "flex",
          gap: 16,
          marginBottom: 12,
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        {/* Scale Mode Toggle */}
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          <span style={{ color: "#96b4d2", fontSize: 12 }}>Scale:</span>
          <div
            style={{
              display: "flex",
              background: "#0d1420",
              borderRadius: 6,
              border: "1px solid #1b2433",
              overflow: "hidden",
            }}
          >
            <button
              type="button"
              onClick={() => setScaleMode("linear")}
              style={{
                padding: "4px 12px",
                fontSize: 12,
                border: "none",
                background: scaleMode === "linear" ? "#ff9b45" : "transparent",
                color: scaleMode === "linear" ? "#08131f" : "#d9e8f4",
                cursor: "pointer",
                fontWeight: scaleMode === "linear" ? 600 : 400,
              }}
            >
              Linear
            </button>
            <button
              type="button"
              onClick={() => setScaleMode("dB")}
              style={{
                padding: "4px 12px",
                fontSize: 12,
                border: "none",
                background: scaleMode === "dB" ? "#ff9b45" : "transparent",
                color: scaleMode === "dB" ? "#08131f" : "#d9e8f4",
                cursor: "pointer",
                fontWeight: scaleMode === "dB" ? 600 : 400,
              }}
            >
              dB
            </button>
          </div>
        </div>

        {/* S-Parameter Multi-select */}
        {numPorts > 0 && (
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ color: "#96b4d2", fontSize: 12 }}>S-Params ({numPorts} ports):</span>
            {allSParams.map(({ key, label }, index) => {
              const isSelected = selectedSParams.has(key)
              const color = COLORS[index % COLORS.length]
              return (
                <label
                  key={key}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                    cursor: "pointer",
                    fontSize: 12,
                    color: isSelected ? "#d9e8f4" : "#5a7a9a",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => toggleSParam(key)}
                    style={{ cursor: "pointer" }}
                  />
                  <span
                    style={{
                      width: 12,
                      height: 2,
                      background: color,
                      display: "inline-block",
                    }}
                  />
                  {label}
                </label>
              )
            })}
          </div>
        )}
      </div>

      {/* Plot */}
      <div style={{ height: containerHeight, minHeight: 200 }}>
        {hasData ? (
          <Plot
            data={plotData}
            layout={layout}
            config={config}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
          />
        ) : (
          <div
            style={{
              height: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#5a7a9a",
              fontSize: 14,
              background: "#0d1420",
              borderRadius: 8,
            }}
          >
            {data === null
              ? "Simulating..."
              : "No simulation data available. Build a circuit with ports to see S-parameters."}
          </div>
        )}
      </div>
    </div>
  )
}
