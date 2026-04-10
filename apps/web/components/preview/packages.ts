import { PREVIEW_PACKAGE_ROUTE_PATH } from "./config";

type PreviewPackageDefinition = {
  defaultFile: string;
  filePaths: Set<string>;
  packageName: string;
};

export type ResolvedPreviewPackageSpecifier = {
  packageName: string;
  relativePath: string;
};

const PREVIEW_PACKAGES = [
  {
    defaultFile: "manifold.js",
    filePaths: new Set(["lib/wasm.js", "manifold.js", "manifold.wasm"]),
    packageName: "manifold-3d",
  },
] satisfies PreviewPackageDefinition[];

const previewPackagesByName = new Map(PREVIEW_PACKAGES.map((entry) => [entry.packageName, entry]));

export function buildPreviewPackageModuleUrl(origin: string, specifier: string) {
  const resolved = resolvePreviewPackageSpecifier(specifier);
  if (!resolved) {
    return undefined;
  }

  const encodedSegments = [resolved.packageName, ...resolved.relativePath.split("/").filter(Boolean)].map((segment) => encodeURIComponent(segment));
  return new URL(`${PREVIEW_PACKAGE_ROUTE_PATH}/${encodedSegments.join("/")}`, origin).toString();
}

export function getPreviewPackageDefinition(packageName: string) {
  return previewPackagesByName.get(packageName);
}

export function resolvePreviewPackageSpecifier(specifier: string): ResolvedPreviewPackageSpecifier | undefined {
  const normalized = specifier.replace(/\\/g, "/");

  for (const definition of PREVIEW_PACKAGES) {
    if (normalized === definition.packageName) {
      return { packageName: definition.packageName, relativePath: definition.defaultFile };
    }
    if (!normalized.startsWith(`${definition.packageName}/`)) {
      continue;
    }

    const relativePath = normalized.slice(definition.packageName.length + 1);
    if (definition.filePaths.has(relativePath)) {
      return { packageName: definition.packageName, relativePath };
    }
  }

  return undefined;
}
