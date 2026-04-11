import { type RefObject, useEffect } from "react";

function getToolCallParts(toolMessage: HTMLElement) {
  const card = toolMessage.firstElementChild;
  if (!(card instanceof HTMLElement)) {
    return null;
  }

  const content = card.firstElementChild;
  if (!(content instanceof HTMLElement) || content.querySelector("[data-pi-tool-call-body]")) {
    return null;
  }

  const children = Array.from(content.children).filter((child): child is HTMLElement => child instanceof HTMLElement);
  if (children.length < 2) {
    return null;
  }

  const [header, ...details] = children;
  if (header.tagName === "BUTTON" || header.querySelector("[data-pi-tool-call-chevron]")) {
    return null;
  }

  return { content, header, details };
}

function setExpanded(toolMessage: HTMLElement, header: HTMLElement, body: HTMLDivElement, expanded: boolean) {
  toolMessage.dataset.piToolCallExpanded = expanded ? "true" : "false";
  header.setAttribute("aria-expanded", expanded ? "true" : "false");
  body.style.display = expanded ? "grid" : "none";
  const chevron = header.querySelector<HTMLElement>("[data-pi-tool-call-chevron]");
  if (chevron) {
    chevron.textContent = expanded ? "v" : ">";
  }
}

function enhanceToolMessage(toolMessage: HTMLElement) {
  const parts = getToolCallParts(toolMessage);
  if (!parts) {
    return;
  }

  const { content, header, details } = parts;
  content.classList.remove("space-y-3");
  content.style.display = "flex";
  content.style.flexDirection = "column";
  content.style.gap = "0.75rem";
  content.style.minWidth = "0";

  const body = document.createElement("div");
  body.dataset.piToolCallBody = "true";
  body.style.display = "none";
  body.style.maxWidth = "100%";
  body.style.minWidth = "0";
  body.style.overflow = "auto";
  body.style.rowGap = "0.75rem";

  for (const detail of details) {
    body.appendChild(detail);
  }

  const chevron = document.createElement("span");
  chevron.dataset.piToolCallChevron = "true";
  chevron.setAttribute("aria-hidden", "true");
  chevron.style.marginLeft = "auto";
  chevron.style.paddingLeft = "0.5rem";
  chevron.style.flexShrink = "0";
  chevron.textContent = ">";

  header.setAttribute("role", "button");
  header.setAttribute("tabindex", "0");
  header.style.cursor = "pointer";
  header.style.userSelect = "none";
  header.appendChild(chevron);

  const toggle = () => {
    const nextExpanded = toolMessage.dataset.piToolCallExpanded !== "true";
    setExpanded(toolMessage, header, body, nextExpanded);
  };

  header.addEventListener("click", toggle);
  header.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggle();
    }
  });

  content.appendChild(body);
  setExpanded(toolMessage, header, body, toolMessage.dataset.piToolCallExpanded === "true");
}

function enhanceToolMessages(container: HTMLElement) {
  const toolMessages = container.querySelectorAll<HTMLElement>("tool-message");
  for (const toolMessage of toolMessages) {
    enhanceToolMessage(toolMessage);
  }
}

export function usePiToolCallCollapse(ref: RefObject<HTMLElement | null>) {
  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }

    const scan = () => enhanceToolMessages(element);
    scan();

    const observer = new MutationObserver(scan);
    observer.observe(element, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, [ref]);
}
