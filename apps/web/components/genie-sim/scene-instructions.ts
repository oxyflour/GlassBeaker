import type { useFrontendTool } from "@copilotkit/react-core";

type ToolParameter = {
  description: string;
  name: string;
  required: boolean;
  type: "array" | "boolean" | "number" | "object" | "string";
};

export const GENIE_SIM_AGENT_NAME = "genie_sim";

const SCENE_TEMPLATE = [
  "from helper import *",
  "",
  "def place_on_top(obj: Shape, support: Shape, xy: tuple[float, float], gap: float = 0.0) -> Shape:",
  '    support_info = get_object_info(support)',
  '    obj_info = get_object_info(obj)',
  '    target_xyz = np.array((xy[0], xy[1], support_info[\"max\"][2] - obj_info[\"min\"][2] + gap))',
  '    delta = target_xyz - np.array(obj_info[\"center\"])',
  '    return transform_shape(obj, translation_matrix(delta))',
  "",
  "def yaw_about_center(shape: Shape, radians: float) -> Shape:",
  '    center = compute_shape_center(shape)',
  '    return transform_shape(shape, rotation_matrix(radians, (0, 0, 1), center))',
  "",
  "@register()",
  "def root_scene() -> Shape:",
  '    table = library_call("usd", oid="table_000", keywords=["table", "workspace_table", "white"] )',
  '    mug = library_call("usd", oid="benchmark_mug_001", keywords=["mug", "cup", "ceramic", "left"] )',
  "    mug = place_on_top(mug, table, xy=(0.0, 0.25), gap=0.0)",
  "    mug = yaw_about_center(mug, radians=0.35)",
  "    return concat_shapes(table, mug)",
].join("\n");

const REGISTERED_GROUP_EXAMPLE = [
  "from helper import *",
  "",
  "def support_top_z(shape: Shape) -> float:",
  '    return float(get_object_info(shape)[\"max\"][2])',
  "",
  "@register()",
  "def place_asset(oid: str, tags: list[str], xyz: tuple[float, float, float], yaw: float = 0.0) -> Shape:",
  '    asset = library_call("usd", oid=oid, keywords=tags)',
  "    asset = transform_shape(asset, translation_matrix(xyz))",
  "    if yaw != 0.0:",
  "        center = compute_shape_center(asset)",
  "        asset = transform_shape(asset, rotation_matrix(yaw, (0, 0, 1), center))",
  "    return asset",
  "",
  "@register()",
  "def root_scene() -> Shape:",
  '    table = library_call("usd", oid="table_000", keywords=["table", "workspace_table", "white"] )',
  '    mug = library_call("place_asset", oid="benchmark_mug_001", tags=["mug", "cup", "ceramic"], xyz=(0.0, 0.25, support_top_z(table)))',
  "    return concat_shapes(table, mug)",
].join("\n");

export const SCENE_ADDITIONAL_INSTRUCTIONS = [
  "You are a Genie Sim scene generation agent.",
  "You turn the user's natural-language scene request into Python code that runs with deps/genie_sim helper.py.",
  "",
  "Workflow:",
  "1. Search assets first with the search_assets tool unless the user already gave exact asset ids.",
  "2. Use the returned asset ids with library_call(\"usd\", oid=...).",
  "3. Output valid helper.py scene code and then call generate_scene with the full code.",
  "",
  "Hard rules:",
  "- The only allowed scene helper import is exactly: from helper import *",
  "- Never import from genie_sim_helper, genie_sim_open, geniesim_helper, geniesim_open, or any other helper alias.",
  "- Only use APIs that helper.py exposes.",
  "- Scene coordinates are +x forward, +y left, +z up. The ground plane is z=0.",
  "- Always define @register() def root_scene() -> Shape.",
  "- Never use undefined type names like Scene, Object, Pose, Vector3, or List[Scene]. If unsure, omit the annotation.",
  "- Never call .add(...) on keywords, Shape, or any Python list. keywords should stay plain lists.",
  '- Call extra registered scene functions via library_call("function_name", ...).',
  "- Never put @register() on helpers that do not return Shape.",
  "- Small math helpers can stay as plain Python functions and be called directly.",
  "- Use modular registered functions when the scene has multiple semantic groups.",
  "- Place objects on supporting surfaces and avoid obvious interpenetration.",
  "- Prefer get_object_info(...)[\"max\"][2] and get_object_info(...)[\"min\"][2] to compute stacking height.",
  "- If you rotate an object after creating it, rotate around compute_shape_center(shape), then translate or re-check height.",
  "- Do not output Markdown fences or prose before calling generate_scene.",
  "- Keep the final response focused on scene generation only.",
  "",
  "Copy this skeleton and fill in asset ids, keywords, positions, and support relationships:",
  SCENE_TEMPLATE,
  "",
  "Registered group pattern:",
  REGISTERED_GROUP_EXAMPLE,
].join("\n");

export const SEARCH_ASSETS_DESCRIPTION =
  "Search Genie Sim assets by semantic description or exact asset id before writing scene code.";

export const SET_SCENE_DESCRIPTION =
  "Execute Genie Sim helper.py scene code and return a structured preview scene.";

const SEARCH_ASSETS_PARAMETERS: ToolParameter[] = [
  {
    description: "English asset query such as 'white dining table', 'red mug', or an exact asset id.",
    name: "query",
    required: true,
    type: "string",
  },
  {
    description: "Maximum number of asset candidates to return.",
    name: "top_k",
    required: false,
    type: "number",
  },
];

const GENERATE_SCENE_PARAMETERS: ToolParameter[] = [
  {
    description: "Full Python code for helper.py, including root_scene().",
    name: "code",
    required: true,
    type: "string",
  },
];

type CopilotParameters = Parameters<typeof useFrontendTool>[0]["parameters"];

export const SEARCH_ASSETS_PARAMETERS_CPK =
  SEARCH_ASSETS_PARAMETERS as unknown as CopilotParameters;
export const SCENE_PARAMETERS_CPK =
  GENERATE_SCENE_PARAMETERS as unknown as CopilotParameters;
