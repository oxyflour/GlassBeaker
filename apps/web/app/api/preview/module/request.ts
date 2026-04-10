import { existsSync } from "node:fs";
import path from "node:path";

import { findPreviewLibraryComponent, PREVIEW_LIBRARY_DEP_SEGMENT, type PreviewLibraryComponent } from "../../../../components/preview/library";

export class PreviewModuleError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PreviewModuleError";
  }
}

export type ResolvedPreviewModuleRequest = {
  component: PreviewLibraryComponent;
  filePath: string;
  relativePath: string;
  rootDir: string;
};

const WEB_DIR = resolveWebDir();

export function resolvePreviewModuleRequest(segments: string[] | undefined): ResolvedPreviewModuleRequest {
  if (!segments || segments.length === 0) {
    throw new PreviewModuleError("Preview module specifier is required.");
  }

  const dependencyIndex = segments.indexOf(PREVIEW_LIBRARY_DEP_SEGMENT);
  const componentSpecifier = (dependencyIndex === -1 ? segments : segments.slice(0, dependencyIndex)).join("/");
  const component = findPreviewLibraryComponent(componentSpecifier);
  if (!component) {
    throw new PreviewModuleError(`Preview library component is not whitelisted: ${componentSpecifier}`);
  }

  const entryPath = path.join(WEB_DIR, component.entryPath);
  const rootDir = path.dirname(entryPath);
  const relativePath = dependencyIndex === -1 ? toPosixPath(path.basename(entryPath)) : normalizeDependencyPath(segments.slice(dependencyIndex + 1));
  const filePath = dependencyIndex === -1 ? entryPath : path.resolve(rootDir, relativePath);

  ensureWithinRoot(rootDir, filePath);
  if (!existsSync(filePath)) {
    throw new PreviewModuleError(`Preview component module not found: ${componentSpecifier}${dependencyIndex === -1 ? "" : ` -> ${relativePath}`}`);
  }

  return { component, filePath, relativePath: toPosixPath(path.relative(rootDir, filePath)), rootDir };
}

function ensureWithinRoot(rootDir: string, targetPath: string) {
  const relativePath = path.relative(rootDir, targetPath);
  if (relativePath.startsWith("..") || path.isAbsolute(relativePath)) {
    throw new PreviewModuleError(`Preview component dependency escapes its allowed directory: ${targetPath}`);
  }
}

function normalizeDependencyPath(segments: string[]) {
  const normalized = segments.join("/").replace(/\\/g, "/");
  if (!normalized || normalized.startsWith("/") || normalized.split("/").some((segment) => segment === "..")) {
    throw new PreviewModuleError(`Invalid preview dependency path: ${normalized || "(empty)"}`);
  }
  return normalized;
}

function resolveWebDir() {
  const cwd = process.cwd();
  if (existsSync(path.join(cwd, "app")) && existsSync(path.join(cwd, "components"))) {
    return cwd;
  }

  return path.join(cwd, "apps", "web");
}

function toPosixPath(filePath: string) {
  return filePath.replace(/\\/g, "/");
}
