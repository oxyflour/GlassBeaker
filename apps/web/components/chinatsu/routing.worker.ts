import { buildRoutes } from "./routing"
import type { CircuitData, Point, RoutedLink } from "./types"

export interface BuildRoutesMessage {
  id: number
  type: "buildRoutes"
  data: CircuitData
  positions: Record<string, Point>
}

export interface BuildRoutesResult {
  id: number
  type: "buildRoutes"
  routes: RoutedLink[]
}

self.onmessage = (event: MessageEvent<BuildRoutesMessage>) => {
  const { id, type, data, positions } = event.data
  if (type === "buildRoutes") {
    const routes = buildRoutes(data, positions)
    self.postMessage({ id, type: "buildRoutes", routes } satisfies BuildRoutesResult)
  }
}
