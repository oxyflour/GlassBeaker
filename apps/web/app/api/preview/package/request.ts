import { existsSync } from "node:fs";
import path from "node:path";

import { getPreviewPackageDefinition } from "../../../../components/preview/packages";

export class PreviewPackageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PreviewPackageError";
  }
}

export type ResolvedPreviewPackageRequest = {
  contentType: string;
  filePath: string;
  packageName: string;
  relativePath: string;
};

const WEB_DIR = resolveWebDir();

export function resolvePreviewPackageRequest(segments: string[] | undefined): ResolvedPreviewPackageRequest {
  if (!segments || segments.length < 2) {
    throw new PreviewPackageError("Preview package specifier is required.");
  }

  const packageName = segments[0];
  const definition = getPreviewPackageDefinition(packageName);
  if (!definition) {
    throw new PreviewPackageError(`Preview package is not whitelisted: ${packageName}`);
  }

  const relativePath = normalizeRelativePath(segments.slice(1));
  if (!definition.filePaths.has(relativePath)) {
    throw new PreviewPackageError(`Preview package file is not whitelisted: ${packageName}/${relativePath}`);
  }

  const packageRoot = path.join(WEB_DIR, "node_modules", packageName);
  const filePath = path.resolve(packageRoot, relativePath);
  ensureWithinRoot(packageRoot, filePath);
  if (!existsSync(filePath)) {
    throw new PreviewPackageError(`Preview package file not found: ${packageName}/${relativePath}`);
  }

  return { contentType: resolveContentType(relativePath), filePath, packageName, relativePath };
}

function ensureWithinRoot(rootDir: string, targetPath: string) {
  const relativePath = path.relative(rootDir, targetPath);
  if (relativePath.startsWith("..") || path.isAbsolute(relativePath)) {
    throw new PreviewPackageError(`Preview package file escapes its package root: ${targetPath}`);
  }
}

function normalizeRelativePath(segments: string[]) {
  const normalized = segments.join("/").replace(/\\/g, "/");
  if (!normalized || normalized.startsWith("/") || normalized.split("/").some((segment) => segment === "..")) {
    throw new PreviewPackageError(`Invalid preview package path: ${normalized || "(empty)"}`);
  }
  return normalized;
}

function resolveContentType(relativePath: string) {
  if (relativePath.endsWith(".wasm")) {
    return "application/wasm";
  }
  if (relativePath.endsWith(".js")) {
    return "application/javascript; charset=utf-8";
  }
  throw new PreviewPackageError(`Unsupported preview package file type: ${relativePath}`);
}

function resolveWebDir() {
  const cwd = process.cwd();
  if (existsSync(path.join(cwd, "app")) && existsSync(path.join(cwd, "components"))) {
    return cwd;
  }

  return path.join(cwd, "apps", "web");
}
