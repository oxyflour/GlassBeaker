import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";

import { init, parse } from "es-module-lexer";
import ts from "typescript";

import { resolvePreviewEsmBaseUrl } from "../../../../components/agent/preview/config";
import { buildPreviewLibraryModuleUrl, findPreviewLibraryComponent, isPreviewLibrarySpecifier } from "../../../../components/agent/preview/library";
import {
  isAliasSpecifier,
  isBareSpecifier,
  isCssModulePath,
  isNodeBuiltinSpecifier,
  isProtocolSpecifier,
  isScriptPath,
  isWorkspaceSpecifier,
  PREVIEW_SUPPORTED_EXTENSIONS,
} from "../../../../components/agent/preview/paths";
import { PreviewModuleError, type ResolvedPreviewModuleRequest } from "./request";

const lexerReady = Promise.resolve(init).then(() => undefined);

export async function buildPreviewModuleSource(request: ResolvedPreviewModuleRequest, origin: string) {
  const source = await readFile(request.filePath, "utf8");
  const code = await buildModuleCode(request, source);
  return await rewriteImportSpecifiers(code, request, origin);
}

async function buildModuleCode(request: ResolvedPreviewModuleRequest, source: string) {
  if (hasUseServerDirective(source)) {
    throw new PreviewModuleError(`Preview component modules cannot use "use server": ${request.component.specifier}`);
  }
  if (request.filePath.endsWith(".json")) {
    return createJsonModule(request.relativePath, source);
  }
  if (request.filePath.endsWith(".css")) {
    if (isCssModulePath(request.filePath)) {
      throw new PreviewModuleError(`CSS Modules are not supported for preview components: ${request.relativePath}`);
    }
    return createCssModule(request.relativePath, source);
  }
  if (!isScriptPath(request.filePath)) {
    throw new PreviewModuleError(`Unsupported preview component file type: ${request.relativePath}`);
  }

  const transpiled = ts.transpileModule(source, {
    compilerOptions: {
      allowJs: true,
      esModuleInterop: true,
      jsx: ts.JsxEmit.ReactJSX,
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
    fileName: request.filePath,
  }).outputText;

  return request.relativePath === path.basename(request.component.entryPath) && request.component.exportName !== "default"
    ? `${transpiled}\nexport { ${request.component.exportName} as default };\n`
    : transpiled;
}

async function rewriteImportSpecifiers(code: string, request: ResolvedPreviewModuleRequest, origin: string) {
  await lexerReady;
  const [imports] = parse(code);
  let rewritten = "";
  let cursor = 0;

  for (const entry of imports) {
    if (entry.n == null) {
      if (entry.d > -1) {
        throw new PreviewModuleError(`Dynamic imports must use string literals in ${request.component.specifier}.`);
      }
      continue;
    }

    rewritten += code.slice(cursor, entry.s);
    rewritten += resolveImportSpecifier(entry.n, request, origin);
    cursor = entry.e;
  }

  return `${rewritten}${code.slice(cursor)}`;
}

function createCssModule(filePath: string, source: string) {
  return [
    `const filePath = ${JSON.stringify(filePath)};`,
    `const css = ${JSON.stringify(source)};`,
    `let style = document.head.querySelector(\`style[data-preview-file=\${JSON.stringify(filePath)}]\`);`,
    "if (!style) {",
    '  style = document.createElement("style");',
    '  style.setAttribute("data-preview-file", filePath);',
    "  document.head.appendChild(style);",
    "}",
    "style.textContent = css;",
    "export default css;",
  ].join("\n");
}

function createJsonModule(filePath: string, source: string) {
  try {
    return `export default ${JSON.stringify(JSON.parse(source))};\n`;
  } catch {
    throw new PreviewModuleError(`Invalid JSON preview component file: ${filePath}`);
  }
}

function hasUseServerDirective(source: string) {
  return /^\s*["']use server["'];?/m.test(source);
}

function resolveImportSpecifier(specifier: string, request: ResolvedPreviewModuleRequest, origin: string) {
  if (specifier === "server-only" || specifier.startsWith("next/")) {
    throw new PreviewModuleError(`Next.js-only imports are not supported in preview components: ${specifier}`);
  }
  if (isProtocolSpecifier(specifier)) {
    throw new PreviewModuleError(`Remote URL imports are not supported in preview components: ${specifier}`);
  }
  if (isNodeBuiltinSpecifier(specifier)) {
    throw new PreviewModuleError(`Node builtins are not supported in preview components: ${specifier}`);
  }
  if (isAliasSpecifier(specifier)) {
    throw new PreviewModuleError(`Repo aliases are not supported in preview components: ${specifier}`);
  }
  if (isWorkspaceSpecifier(specifier)) {
    throw new PreviewModuleError(`Workspace packages are not supported in preview components: ${specifier}`);
  }
  if (isPreviewLibrarySpecifier(specifier)) {
    if (!findPreviewLibraryComponent(specifier)) {
      throw new PreviewModuleError(`Preview library component is not whitelisted: ${specifier}`);
    }
    return buildPreviewLibraryModuleUrl(origin, specifier);
  }
  if (isBareSpecifier(specifier)) {
    return `${resolvePreviewEsmBaseUrl()}/${specifier}`;
  }
  if (specifier.startsWith("/")) {
    return specifier;
  }

  return buildPreviewLibraryModuleUrl(origin, request.component.specifier, resolveRelativeImport(request, specifier));
}

function resolveRelativeImport(request: ResolvedPreviewModuleRequest, specifier: string) {
  const basePath = path.resolve(path.dirname(request.filePath), specifier);
  const candidates = PREVIEW_SUPPORTED_EXTENSIONS.some((extension) => basePath.endsWith(extension))
    ? [basePath]
    : [
      ...PREVIEW_SUPPORTED_EXTENSIONS.map((extension) => `${basePath}${extension}`),
      ...PREVIEW_SUPPORTED_EXTENSIONS.map((extension) => path.join(basePath, `index${extension}`)),
    ];

  for (const candidate of candidates) {
    const normalizedCandidate = path.normalize(candidate);
    const relativePath = path.relative(request.rootDir, normalizedCandidate);
    if (!relativePath.startsWith("..") && !path.isAbsolute(relativePath) && existsSync(normalizedCandidate)) {
      return relativePath.replace(/\\/g, "/");
    }
  }

  throw new PreviewModuleError(`Cannot resolve preview component import "${specifier}" from ${request.component.specifier}.`);
}
