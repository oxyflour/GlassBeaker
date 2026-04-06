import type { useFrontendTool } from "@copilotkit/react-core";

type PreviewToolParameter = {
  description: string;
  name: string;
  required: boolean;
  type: "array" | "boolean" | "number" | "object" | "string";
};

const PREVIEW_RULES = [
  "You are a professional React developer who can build browser-only applications.",
  "Output front-end code only. Do not emit npm commands.",
  "Only use relative file imports or bare npm package imports.",
  "Every relative import used by `entry` or helper modules must have a matching file in `files`.",
  "When styling is needed, create plain `.css` files in `files` and import them from `entry` or child components.",
  "Do not use Node APIs, repo aliases, package.json files, CSS Modules, images, fonts, or binary assets.",
  "Use `set_app_code` to create or replace the live preview.",
  "Use `set_app_props` when only the props should change.",
];

export const PREVIEW_ADDITIONAL_INSTRUCTIONS = PREVIEW_RULES.join("\n");

export const PREVIEW_SET_APP_CODE_DESCRIPTION =
  "Create or replace the live React preview. `entry` must default export a React component. Put every imported relative module, helper component, and plain `.css` file in `files` and import them explicitly.";

export const PREVIEW_SET_APP_PROPS_DESCRIPTION =
  "Update only the JSON props for the current preview without replacing the component code.";

export const PREVIEW_CODE_PARAMETERS: PreviewToolParameter[] = [
  { name: "entry", type: "string", description: "React component source code that default exports a component.", required: true },
  { name: "props", type: "object", description: "JSON props passed into the generated component.", required: true },
  { name: "files", type: "object", description: "Additional files with file path as key and file content as value. Use this for helper components and plain `.css` files.", required: true },
];

export const PREVIEW_PROPS_PARAMETERS: PreviewToolParameter[] = [
  { name: "props", type: "object", description: "The full JSON props object passed to the current preview component.", required: true },
];

type CopilotParameters = Parameters<typeof useFrontendTool>[0]["parameters"];

export const PREVIEW_CODE_PARAMETERS_CPK = PREVIEW_CODE_PARAMETERS as unknown as CopilotParameters;
export const PREVIEW_PROPS_PARAMETERS_CPK = PREVIEW_PROPS_PARAMETERS as unknown as CopilotParameters;

export function buildPreviewSystemPrompt(props: unknown) {
  return [...PREVIEW_RULES, `Current preview props: ${JSON.stringify(props, null, 2)}`].join("\n\n");
}
