import { buildPreviewLibraryModuleUrl, findPreviewLibraryComponent, isPreviewLibrarySpecifier } from "./library";
import { buildPreviewPackageModuleUrl } from "./packages";
import type { PreviewFiles } from "./state";
import {
  isAliasSpecifier,
  isBareSpecifier,
  isCssModulePath,
  isNodeBuiltinSpecifier,
  isProtocolSpecifier,
  isScriptPath,
  isWorkspaceSpecifier,
  joinEsmSpecifierUrl,
  resolveVirtualPath,
  toVirtualModuleUrl,
} from "./paths";

let babelPromise: Promise<typeof import("@babel/standalone")> | null = null;
let lexerPromise: Promise<typeof import("es-module-lexer")> | null = null;

export type PreviewModule = {
  code: string;
  dependencies: string[];
  validationImports: string[];
};

export async function buildPreviewModule(
  filePath: string,
  source: string,
  files: PreviewFiles,
  esmBaseUrl: string,
  previewOrigin: string,
): Promise<PreviewModule> {
  if (filePath.endsWith(".json")) {
    return { code: createJsonModule(filePath, source), dependencies: [], validationImports: [] };
  }
  if (filePath.endsWith(".css")) {
    if (isCssModulePath(filePath)) {
      throw new Error(`CSS Modules are not supported in preview: ${filePath}`);
    }
    return { code: createCssModule(filePath, source), dependencies: [], validationImports: [] };
  }
  if (!isScriptPath(filePath)) {
    throw new Error(`Unsupported preview file type: ${filePath}`);
  }

  const transpiled = await transpileScriptModule(filePath, source);
  return await rewriteModuleSpecifiers(filePath, transpiled, files, esmBaseUrl, previewOrigin);
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
    throw new Error(`Invalid JSON preview file: ${filePath}`);
  }
}

async function rewriteModuleSpecifiers(
  filePath: string,
  code: string,
  files: PreviewFiles,
  esmBaseUrl: string,
  previewOrigin: string,
): Promise<PreviewModule> {
  const { parse } = await loadLexer();
  const [imports] = parse(code);
  const dependencies: string[] = [];
  const validationImports: string[] = [];

  if (imports.length === 0) {
    return { code, dependencies, validationImports };
  }

  let rewritten = "";
  let cursor = 0;
  for (const entry of imports) {
    if (entry.n == null) {
      if (entry.d > -1) {
        throw new Error(`Dynamic imports must use string literals in ${filePath}.`);
      }
      continue;
    }

    const replacement = resolvePreviewSpecifier(filePath, entry.n, files, esmBaseUrl, previewOrigin);
    rewritten += code.slice(cursor, entry.s);
    rewritten += formatResolvedSpecifier(replacement.specifier, entry.d > -1);
    cursor = entry.e;
    if (replacement.internalPath) {
      dependencies.push(replacement.internalPath);
    }
    if (replacement.validationImport) {
      validationImports.push(replacement.validationImport);
    }
  }

  rewritten += code.slice(cursor);
  return {
    code: rewritten,
    dependencies: Array.from(new Set(dependencies)),
    validationImports: Array.from(new Set(validationImports)),
  };
}

function formatResolvedSpecifier(specifier: string, isDynamicImport: boolean) {
  return isDynamicImport ? JSON.stringify(specifier) : specifier;
}

function resolvePreviewSpecifier(
  filePath: string,
  specifier: string,
  files: PreviewFiles,
  esmBaseUrl: string,
  previewOrigin: string,
) {
  if (isProtocolSpecifier(specifier)) {
    throw new Error(`Remote URLs are not supported in preview imports: ${specifier}`);
  }
  if (isNodeBuiltinSpecifier(specifier)) {
    throw new Error(`Node builtins are not supported in preview imports: ${specifier}`);
  }
  if (isAliasSpecifier(specifier)) {
    throw new Error(`Repo aliases are not supported in preview imports: ${specifier}`);
  }
  if (isWorkspaceSpecifier(specifier)) {
    throw new Error(`Workspace packages are not supported in preview imports: ${specifier}`);
  }
  if (isPreviewLibrarySpecifier(specifier)) {
    if (!findPreviewLibraryComponent(specifier)) {
      throw new Error(`Preview library component is not whitelisted: ${specifier}`);
    }
    const url = buildPreviewLibraryModuleUrl(previewOrigin, specifier);
    return { specifier: url, validationImport: url };
  }
  const previewPackageUrl = buildPreviewPackageModuleUrl(previewOrigin, specifier);
  if (previewPackageUrl) {
    return { specifier: previewPackageUrl };
  }
  if (isBareSpecifier(specifier)) {
    return { specifier: joinEsmSpecifierUrl(esmBaseUrl, specifier) };
  }

  const resolvedPath = resolveVirtualPath(filePath, specifier, files);
  if (!resolvedPath) {
    throw new Error(`Cannot resolve preview import "${specifier}" from ${filePath}.`);
  }
  if (isCssModulePath(resolvedPath)) {
    throw new Error(`CSS Modules are not supported in preview imports: ${specifier}`);
  }
  return { internalPath: resolvedPath, specifier: toVirtualModuleUrl(resolvedPath) };
}

async function transpileScriptModule(filePath: string, source: string) {
  const Babel = await loadBabel();
  const isTypeScript = filePath.endsWith(".ts") || filePath.endsWith(".tsx");
  const isJsx = filePath.endsWith(".jsx") || filePath.endsWith(".tsx");
  const presets: any[] = [];

  if (isTypeScript) {
    presets.push(["typescript", { allExtensions: true, isTSX: isJsx }]);
  }
  if (isJsx || filePath.endsWith(".js") || filePath.endsWith(".ts")) {
    presets.push(["react", { runtime: "automatic" }]);
  }

  return (
    Babel.transform(source, {
      comments: false,
      filename: filePath,
      presets,
      retainLines: true,
      sourceType: "module",
    }).code ?? source
  );
}

async function loadBabel() {
  babelPromise ??= import("@babel/standalone");
  return await babelPromise;
}

async function loadLexer() {
  lexerPromise ??= import("es-module-lexer");
  const lexer = await lexerPromise;
  await lexer.init;
  return lexer;
}
