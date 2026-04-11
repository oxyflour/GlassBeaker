import { PALETTE_ITEMS } from "./defaults"
import { type CircuitBlockType } from "./types"

export function CircuitToolbar({
  canDelete,
  canRotate,
  canZoomIn,
  canZoomOut,
  onAdd,
  onDelete,
  onRotate,
  onZoomIn,
  onZoomOut,
  onZoomReset,
  zoomLabel,
}: {
  canDelete: boolean
  canRotate: boolean
  canZoomIn: boolean
  canZoomOut: boolean
  onAdd: (type: CircuitBlockType) => void
  onDelete: () => void
  onRotate: () => void
  onZoomIn: () => void
  onZoomOut: () => void
  onZoomReset: () => void
  zoomLabel: string
}) {
  return (
    <>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(11, minmax(0, 1fr))", gap: 6, padding: 8, borderBottom: "1px solid #222c3f", alignItems: "stretch" }}>
        {PALETTE_ITEMS.map((item) => (
          <button key={item.type} type="button" onClick={() => onAdd(item.type)} style={{ border: "1px solid #39465f", background: "#202a3c", color: "#f3f5fa", borderRadius: 0, padding: "6px 8px", cursor: "pointer", textAlign: "left", minWidth: 0 }}>
            <span style={{ display: "block", fontWeight: 700 }}>{item.label}</span>
            <span style={{ display: "block", fontSize: 11, opacity: 0.72 }}>{item.hint}</span>
          </button>
        ))}
        <button type="button" onClick={onRotate} disabled={!canRotate} style={{ border: "1px solid #39465f", background: canRotate ? "#25324a" : "#1a2231", color: "#f3f5fa", borderRadius: 0, padding: "6px 8px", cursor: canRotate ? "pointer" : "not-allowed", textAlign: "left" }}>
          Rotate
        </button>
        <button type="button" onClick={onDelete} disabled={!canDelete} style={{ border: "1px solid #5d3a46", background: canDelete ? "#41212a" : "#1a2231", color: "#f6d9df", borderRadius: 0, padding: "6px 8px", cursor: canDelete ? "pointer" : "not-allowed", textAlign: "left" }}>
          Delete
        </button>
        <button type="button" onClick={onZoomOut} disabled={!canZoomOut} style={{ border: "1px solid #39465f", background: canZoomOut ? "#25324a" : "#1a2231", color: "#f3f5fa", borderRadius: 0, padding: "6px 8px", cursor: canZoomOut ? "pointer" : "not-allowed", textAlign: "left" }}>
          Zoom Out
        </button>
        <button type="button" onClick={onZoomReset} style={{ border: "1px solid #39465f", background: "#25324a", color: "#f3f5fa", borderRadius: 0, padding: "6px 8px", cursor: "pointer", textAlign: "left" }}>
          {zoomLabel}
        </button>
        <button type="button" onClick={onZoomIn} disabled={!canZoomIn} style={{ border: "1px solid #39465f", background: canZoomIn ? "#25324a" : "#1a2231", color: "#f3f5fa", borderRadius: 0, padding: "6px 8px", cursor: canZoomIn ? "pointer" : "not-allowed", textAlign: "left" }}>
          Zoom In
        </button>
      </div>

      <div style={{ padding: "6px 8px 8px", color: "#a8b3c6", fontSize: 12 }}>
        Click a pin to start a route. Drag blank space to box-select. Right or middle drag, hold Space and drag, or use the wheel to pan. Use Ctrl plus wheel, pinch, or the zoom buttons to scale.
      </div>
    </>
  )
}
