import { createPreviewSrcDoc } from "./frame-html";
import { joinEsmSpecifierUrl, normalizePreviewFiles, toVirtualModuleUrl } from "./paths";
import type { PreviewFiles } from "./state";
import { buildPreviewModule } from "./transpile";

const BOOT_FILE = "/__preview_boot__.js";
const ROOT_FILE = "/App.js";

export type CompiledPreview = {
  revoke: () => void;
  srcDoc: string;
  validationImports: string[];
};

export async function compilePreview(files: PreviewFiles, esmBaseUrl: string, previewOrigin: string): Promise<CompiledPreview> {
  const normalizedFiles = normalizePreviewFiles(files);
  const modules = new Map<string, string>();
  const validationImports = new Set<string>();

  await visitModule(ROOT_FILE, normalizedFiles, esmBaseUrl, previewOrigin, modules, validationImports);
  modules.set(BOOT_FILE, createBootModule(esmBaseUrl));

  const urls: string[] = [];
  const imports: Record<string, string> = {};
  for (const [filePath, code] of modules) {
    const url = URL.createObjectURL(new Blob([code], { type: "text/javascript" }));
    urls.push(url);
    imports[toVirtualModuleUrl(filePath)] = url;
  }

  return {
    revoke: () => urls.forEach((url) => URL.revokeObjectURL(url)),
    srcDoc: createPreviewSrcDoc(imports, toVirtualModuleUrl(BOOT_FILE)),
    validationImports: Array.from(validationImports),
  };
}

async function visitModule(
  filePath: string,
  files: PreviewFiles,
  esmBaseUrl: string,
  previewOrigin: string,
  modules: Map<string, string>,
  validationImports: Set<string>,
) {
  if (modules.has(filePath)) {
    return;
  }

  const source = files[filePath];
  if (source === undefined) {
    throw new Error(`Preview entry file is missing: ${filePath}`);
  }

  const module = await buildPreviewModule(filePath, source, files, esmBaseUrl, previewOrigin);
  modules.set(filePath, module.code);
  module.validationImports.forEach((url) => validationImports.add(url));

  for (const dependency of module.dependencies) {
    await visitModule(dependency, files, esmBaseUrl, previewOrigin, modules, validationImports);
  }
}

function createBootModule(esmBaseUrl: string) {
  return [
    `import { StrictMode, createElement } from ${JSON.stringify(joinEsmSpecifierUrl(esmBaseUrl, "react"))};`,
    `import { createRoot } from ${JSON.stringify(joinEsmSpecifierUrl(esmBaseUrl, "react-dom/client"))};`,
    `import App from ${JSON.stringify(toVirtualModuleUrl(ROOT_FILE))};`,
    'const container = document.getElementById("root");',
    'if (!container) throw new Error("Preview root element not found.");',
    "createRoot(container).render(createElement(StrictMode, null, createElement(App)));",
  ].join("\n");
}
