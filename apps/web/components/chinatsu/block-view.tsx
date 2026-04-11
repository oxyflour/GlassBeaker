import { type PointerEvent as ReactPointerEvent } from "react"

import { BLOCK_META } from "./defaults"
import { isPointLinkPin } from "./geometry"
import { getBlockLayout } from "./symbol-geometry"
import { CircuitSymbol, getBlockLabelColor, getBlockLabelStyle } from "./symbols"
import { type Block, type LinkPin, type Point } from "./types"

const LABEL_FONT = "'Segoe UI', 'Noto Sans', sans-serif"

function getLabelLines(block: Block) {
  return [block.label, block.id, ...(block.value?.split("\n") ?? [])].filter(Boolean)
}

export function CircuitBlockView({
  activeSource,
  block,
  draggingId,
  onBeginDrag,
  onPinClick,
  onSelect,
  position,
  selected,
}: {
  activeSource: LinkPin | null
  block: Block
  draggingId: string | null
  onBeginDrag: (id: string, event: ReactPointerEvent<HTMLDivElement>) => void
  onPinClick: (target: LinkPin) => void
  onSelect: (id: string) => void
  position: Point
  selected: boolean
}) {
  const layout = getBlockLayout(block)
  const pinNames = block.pins ?? BLOCK_META[block.type ?? "snp"].pinNames.map((name) => ({ name }))

  return (
    <div style={{ position: "absolute", left: position.x, top: position.y, width: layout.width, height: layout.height }}>
      <div
        onPointerDown={(event) => {
          if (event.button === 0) {
            onBeginDrag(block.id, event)
          }
        }}
        onClick={(event) => {
          event.stopPropagation()
          onSelect(block.id)
        }}
        style={{ width: layout.width, height: layout.height, cursor: draggingId === block.id ? "grabbing" : "grab", outline: selected ? "1px dashed rgba(178, 69, 143, 0.5)" : "none", outlineOffset: 6 }}
      >
        <CircuitSymbol block={block} selected={selected} />
      </div>

      {layout.pinPoints.map((pin, index) => {
        const active = activeSource && !isPointLinkPin(activeSource) && activeSource.node === block.id && activeSource.pin === index
        return (
          <button
            key={`${block.id}:${index}`}
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              onSelect(block.id)
              onPinClick({ node: block.id, pin: index })
            }}
            aria-label={`Connect ${block.id} ${pinNames[index]?.name ?? `pin ${index + 1}`}`}
            style={{
              position: "absolute",
              left: pin.x - 7,
              top: pin.y - 7,
              width: 14,
              height: 14,
              borderRadius: 999,
              border: `2px solid ${active ? "#d8922f" : "#b2458f"}`,
              background: active ? "#fff4dc" : "#ffffff",
              boxSizing: "border-box",
              cursor: "crosshair",
            }}
          />
        )
      })}

      {block.type !== "ground" ? (
        <div style={{ position: "absolute", color: getBlockLabelColor(), fontFamily: LABEL_FONT, fontSize: 14, lineHeight: 1.25, pointerEvents: "none", whiteSpace: "pre", ...getBlockLabelStyle(block) }}>
          {getLabelLines(block).map((line) => (
            <div key={`${block.id}:${line}`}>{line}</div>
          ))}
        </div>
      ) : null}
    </div>
  )
}
