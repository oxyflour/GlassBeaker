import { PREVIEW_MODULE_ROUTE_PATH } from "./config";

export const PREVIEW_LIBRARY_DEP_SEGMENT = "__deps__";
export const PREVIEW_LIBRARY_NAMESPACE = "@glassbeaker-preview";

export type PreviewLibraryComponent = {
  entryPath: string;
  example: string;
  exportName: "default" | string;
  props: string[];
  specifier: string;
  title: string;
  whenToUse: string;
};

export const PREVIEW_LIBRARY_COMPONENTS: PreviewLibraryComponent[] = [
  {
    entryPath: "components/chinatsu/circuit.tsx",
    example: 'import Circuit from "@glassbeaker-preview/chinatsu/Circuit";\n\nexport default function Entry() {\n  return <Circuit />;\n}',
    exportName: "default",
    props: ["data?: CircuitData", "onChange?: (data: CircuitData) => void"],
    specifier: "@glassbeaker-preview/chinatsu/Circuit",
    title: "Circuit",
    whenToUse: "Interactive circuit editor with draggable blocks and editable links.",
  },
  {
    entryPath: "components/nijika/antenna.tsx",
    example: 'import Antenna from "@glassbeaker-preview/nijika/Antenna";\n\nexport default function Entry() {\n  return <div style={{ height: "100vh" }}><Antenna /></div>;\n}',
    exportName: "default",
    props: [],
    specifier: "@glassbeaker-preview/nijika/Antenna",
    title: "Antenna",
    whenToUse: "3D antenna showcase with Three.js and manifold-3d for product-style visualization.",
  },
];

const previewLibraryBySpecifier = new Map(PREVIEW_LIBRARY_COMPONENTS.map((component) => [component.specifier, component]));

export function buildPreviewLibraryCatalogPrompt() {
  return [
    "Whitelisted repo preview components are available under `@glassbeaker-preview/*`. Prefer importing them instead of copying their source into `files`.",
    ...PREVIEW_LIBRARY_COMPONENTS.map(formatPreviewLibraryPromptEntry),
  ].join("\n\n");
}

export function buildPreviewLibraryModuleUrl(origin: string, specifier: string, dependencyPath?: string) {
  const encodedSegments = [
    ...specifier.split("/"),
    ...(dependencyPath ? [PREVIEW_LIBRARY_DEP_SEGMENT, ...dependencyPath.replace(/\\/g, "/").split("/").filter(Boolean)] : []),
  ].map((segment) => encodeURIComponent(segment));

  return new URL(`${PREVIEW_MODULE_ROUTE_PATH}/${encodedSegments.join("/")}`, origin).toString();
}

export function findPreviewLibraryComponent(specifier: string) {
  return previewLibraryBySpecifier.get(specifier);
}

export function getPreviewLibraryReadableCatalog() {
  return PREVIEW_LIBRARY_COMPONENTS.map(({ example, props, specifier, title, whenToUse }) => ({
    example,
    props,
    specifier,
    title,
    whenToUse,
  }));
}

export function isPreviewLibrarySpecifier(specifier: string) {
  return specifier.startsWith(`${PREVIEW_LIBRARY_NAMESPACE}/`);
}

function formatPreviewLibraryPromptEntry(component: PreviewLibraryComponent) {
  const props = component.props.length > 0 ? component.props.join("; ") : "none";
  return [
    `Component: ${component.title}`,
    `Specifier: ${component.specifier}`,
    `Use when: ${component.whenToUse}`,
    `Props: ${props}`,
    `Example:\n${component.example}`,
  ].join("\n");
}
