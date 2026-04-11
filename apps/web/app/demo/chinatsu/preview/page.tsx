"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Group, Panel, Separator } from "react-resizable-panels"
import Circuit from "../../../../components/chinatsu/circuit"
import SParameterChart, { type SParameterData } from "../../../../components/chinatsu/s-parameter-chart"
import type { CircuitData } from "../../../../components/chinatsu/types"

interface SimulationState {
  data: SParameterData | null
  loading: boolean
  error: string | null
}

async function simulateCircuit(
  circuit: CircuitData,
  freqStart: number,
  freqStop: number,
  freqPoints: number,
  signal?: AbortSignal
): Promise<SParameterData> {
  const response = await fetch("/python/chinatsu/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      circuit,
      freq_start: freqStart,
      freq_stop: freqStop,
      freq_points: freqPoints,
    }),
    signal,
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(errorText || `Simulation failed: ${response.status}`)
  }

  return response.json()
}

export default function ChinatsuPreviewPage() {
  const [circuitData, setCircuitData] = useState<CircuitData | null>(null)
  const [simState, setSimState] = useState<SimulationState>({
    data: null,
    loading: false,
    error: null,
  })

  // Frequency settings
  const [freqStart, setFreqStart] = useState(0.1) // GHz
  const [freqStop, setFreqStop] = useState(10) // GHz
  const [freqPoints, setFreqPoints] = useState(1001)

  const abortControllerRef = useRef<AbortController | null>(null)
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null)

  const runSimulation = useCallback(
    async (data: CircuitData) => {
      // Cancel previous request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
      abortControllerRef.current = new AbortController()

      setSimState((prev) => ({ ...prev, loading: true, error: null }))

      try {
        const result = await simulateCircuit(
          data,
          freqStart * 1e9, // Convert to Hz
          freqStop * 1e9,
          freqPoints,
          abortControllerRef.current.signal
        )
        setSimState({ data: result, loading: false, error: null })
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          return // Ignore abort errors
        }
        setSimState({
          data: null,
          loading: false,
          error: error instanceof Error ? error.message : "Unknown error",
        })
      }
    },
    [freqStart, freqStop, freqPoints]
  )

  // Debounced simulation when circuit changes
  useEffect(() => {
    if (!circuitData) return

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
    }

    debounceTimerRef.current = setTimeout(() => {
      runSimulation(circuitData)
    }, 300) // 300ms debounce

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
      }
    }
  }, [circuitData, runSimulation])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
      }
    }
  }, [])

  const handleCircuitChange = useCallback((data: CircuitData) => {
    setCircuitData((prev) => {
      // If this is the first time we receive data, trigger simulation immediately
      if (!prev) {
        runSimulation(data)
      }
      return data
    })
  }, [runSimulation])

  return <Group orientation="horizontal" style={{ height: "100%" }}>
    <Panel defaultSize={60} minSize={30} style={{ overflow: 'hidden' }}>
      <Circuit onChange={handleCircuitChange} />
    </Panel>

    <Separator />

    <Panel defaultSize={40} minSize={25}>
      <SParameterChart
        data={simState.data}
        title="S-Parameters"
        fillContainer
      />
    </Panel>
  </Group>
}
