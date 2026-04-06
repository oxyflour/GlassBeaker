import type { PreviewFiles } from "./state";

export const PREVIEW_VIRTUAL_ORIGIN = "https://preview.local";

const SUPPORTED_EXTENSIONS = [".js", ".jsx", ".ts", ".tsx", ".json", ".css"];
const NODE_BUILTINS = new Set([
  "assert",
  "buffer",
  "child_process",
  "crypto",
  "events",
  "fs",
  "http",
  "https",
  "module",
  "net",
  "os",
  "path",
  "process",
  "stream",
  "timers",
  "tty",
  "url",
  "util",
  "zlib",
]);

export function normalizeFilePath(filePath: string) {
  const withSlashes = filePath.replace(/\\/g, "/");
  const absolutePath = withSlashes.startsWith("/") ? withSlashes : `/${withSlashes}`;
  return new URL(absolutePath, PREVIEW_VIRTUAL_ORIGIN).pathname;
}

export function normalizePreviewFiles(files: PreviewFiles) {
  return Object.fromEntries(Object.entries(files).map(([filePath, content]) => [normalizeFilePath(filePath), content]));
}

export function isAliasSpecifier(specifier: string) {
  return specifier.startsWith("@/") || specifier.startsWith("~/") || specifier.startsWith("#/") || specifier.startsWith("%/");
}

export function isBareSpecifier(specifier: string) {
  return !specifier.startsWith(".") && !specifier.startsWith("/") && !isProtocolSpecifier(specifier);
}

export function isCssModulePath(filePath: string) {
  return filePath.endsWith(".module.css");
}

export function isNodeBuiltinSpecifier(specifier: string) {
  const normalized = specifier.startsWith("node:") ? specifier.slice(5) : specifier;
  const packageName = normalized.startsWith("@") ? normalized.split("/").slice(0, 2).join("/") : normalized.split("/")[0];
  return NODE_BUILTINS.has(packageName);
}

export function isProtocolSpecifier(specifier: string) {
  return /^[a-zA-Z][a-zA-Z\d+.-]*:/.test(specifier);
}

export function isScriptPath(filePath: string) {
  return filePath.endsWith(".js") || filePath.endsWith(".jsx") || filePath.endsWith(".ts") || filePath.endsWith(".tsx");
}

export function isWorkspaceSpecifier(specifier: string) {
  return specifier.startsWith("@glassbeaker/") || specifier.startsWith("glassbeaker-");
}

export function joinEsmSpecifierUrl(baseUrl: string, specifier: string) {
  return `${baseUrl.replace(/\/$/, "")}/${specifier}`;
}

export function toVirtualModuleUrl(filePath: string) {
  return new URL(normalizeFilePath(filePath), PREVIEW_VIRTUAL_ORIGIN).toString();
}

export function resolveVirtualPath(fromFile: string, specifier: string, files: PreviewFiles) {
  const resolvedPath = specifier.startsWith("/")
    ? normalizeFilePath(specifier)
    : normalizeFilePath(new URL(specifier, `${PREVIEW_VIRTUAL_ORIGIN}${fromFile}`).pathname);

  for (const candidate of createPathCandidates(resolvedPath)) {
    if (candidate in files) {
      return candidate;
    }
  }

  return undefined;
}

function createPathCandidates(filePath: string) {
  const candidates = [filePath];
  const hasExtension = SUPPORTED_EXTENSIONS.some((extension) => filePath.endsWith(extension));
  const indexBase = filePath.endsWith("/") ? filePath.slice(0, -1) : filePath;

  if (!hasExtension) {
    for (const extension of SUPPORTED_EXTENSIONS) {
      candidates.push(`${filePath}${extension}`);
      candidates.push(`${indexBase}/index${extension}`);
    }
  }

  return Array.from(new Set(candidates));
}
