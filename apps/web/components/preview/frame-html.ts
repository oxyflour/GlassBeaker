import { FRAME_MESSAGE_SOURCE } from "./messages";

export function createPreviewSrcDoc(importMap: Record<string, string>, bootModuleUrl: string) {
  const escapedImportMap = escapeInlineScript(JSON.stringify({ imports: importMap }));
  const frameSource = JSON.stringify(FRAME_MESSAGE_SOURCE);
  const escapedBootModuleUrl = JSON.stringify(bootModuleUrl);

  return [
    "<!doctype html>",
    '<html lang="en">',
    "<head>",
    '<meta charset="utf-8" />',
    '<meta name="viewport" content="width=device-width, initial-scale=1" />',
    "<style>html,body,#root{height:100%;margin:0;}body{font-family:system-ui,sans-serif;background:#fff;color:#111;}*{box-sizing:border-box;}</style>",
    `<script>`,
    `const source = ${frameSource};`,
    "window.__notifyPreviewFrame = (type, error) => {",
    '  window.parent.postMessage({ source, type, error }, "*");',
    "};",
    "window.addEventListener('error', (event) => {",
    "  window.__notifyPreviewFrame('error', event.error?.stack || event.message || 'Unknown preview error.');",
    "});",
    "window.addEventListener('unhandledrejection', (event) => {",
    "  const reason = event.reason;",
    "  window.__notifyPreviewFrame('error', reason?.stack || reason?.message || String(reason));",
    "});",
    "</script>",
    `<script type="importmap">${escapedImportMap}</script>`,
    "</head>",
    "<body>",
    '<div id="root"></div>',
    '<script type="module">',
    `import(${escapedBootModuleUrl})`,
    ".then(() => window.__notifyPreviewFrame('ready'))",
    ".catch((error) => window.__notifyPreviewFrame('error', error?.stack || error?.message || String(error)));",
    "</script>",
    "</body>",
    "</html>",
  ].join("");
}

function escapeInlineScript(value: string) {
  return value.replace(/</g, "\\u003c");
}
