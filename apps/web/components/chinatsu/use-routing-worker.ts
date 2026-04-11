import { useEffect, useRef, useState } from "react"

import type { CircuitData, Point, RoutedLink } from "./types"

interface WorkerState {
  isCalculating: boolean
  routes: RoutedLink[]
}

interface WorkerMessage {
  id: number
  routes: RoutedLink[]
  type: "buildRoutes"
}

/**
 * 使用 Web Worker 进行异步路由计算，避免阻塞主线程。
 * 适合处理复杂电路（50+ 链接）的 A* 路径查找。
 */
export function useRoutingWorker(data: CircuitData, positions: Record<string, Point>): WorkerState {
  const workerRef = useRef<Worker | null>(null)
  const requestIdRef = useRef(0)
  const pendingRef = useRef<Set<number>>(new Set())
  const [state, setState] = useState<WorkerState>({ routes: [], isCalculating: false })
  const lastValidRoutesRef = useRef<RoutedLink[]>([])

  useEffect(() => {
    let worker: Worker

    try {
      worker = new Worker(new URL("./routing.worker.ts", import.meta.url), { type: "module" })
      workerRef.current = worker

      worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
        const { id, routes } = event.data
        pendingRef.current.delete(id)
        lastValidRoutesRef.current = routes
        setState({ routes, isCalculating: pendingRef.current.size > 0 })
      }

      worker.onerror = (error) => {
        console.error("Routing worker error:", error)
        // 出错时保持上次有效结果
        setState({ routes: lastValidRoutesRef.current, isCalculating: false })
      }
    } catch (error) {
      console.error("Failed to initialize routing worker:", error)
    }

    return () => {
      worker?.terminate()
      workerRef.current = null
    }
  }, [])

  useEffect(() => {
    const worker = workerRef.current
    if (!worker) {
      // Worker 初始化失败时，回退到同步计算
      try {
        const { buildRoutes } = require("./routing")
        const routes = buildRoutes(data, positions)
        lastValidRoutesRef.current = routes
        setState({ routes, isCalculating: false })
      } catch (error) {
        console.error("Fallback routing calculation failed:", error)
      }
      return
    }

    const id = ++requestIdRef.current
    pendingRef.current.add(id)

    // 使用防抖避免频繁计算
    const timer = setTimeout(() => {
      worker.postMessage({ id, type: "buildRoutes", data, positions })
    }, 16) // ~1 帧的延迟

    setState((previous) => ({ ...previous, isCalculating: true }))

    return () => {
      clearTimeout(timer)
      pendingRef.current.delete(id)
    }
  }, [data, positions])

  return state
}
