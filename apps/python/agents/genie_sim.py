import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

SCENE_TEMPLATE = "\n".join(
    [
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
        '    table = library_call("usd", oid="table_000", keywords=["table", "workspace_table", "white"])',
        '    mug = library_call("usd", oid="benchmark_mug_001", keywords=["mug", "cup", "ceramic", "left"])',
        "    mug = place_on_top(mug, table, xy=(0.0, 0.25), gap=0.0)",
        "    mug = yaw_about_center(mug, radians=0.35)",
        "    return concat_shapes(table, mug)",
    ]
)

REGISTERED_HELPER_EXAMPLE = "\n".join(
    [
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
        '    table = library_call("usd", oid="table_000", keywords=["table", "workspace_table", "white"])',
        '    mug = library_call("place_asset", oid="benchmark_mug_001", tags=["mug", "cup", "ceramic"], xyz=(0.0, 0.25, support_top_z(table)))',
        "    return concat_shapes(table, mug)",
    ]
)

GENIE_SIM_AGENT_INSTRUCTIONS = f"""
You are a Genie Sim scene generation agent.
Use the available frontend tools to search assets and generate runnable helper.py scene code.

Workflow:
1. Search assets first unless the user already gave exact asset ids.
2. Use the returned asset ids with library_call("usd", oid=...).
3. Call generate_scene with full helper.py code.
4. If generate_scene fails, read the returned traceback, fix the code, and retry.

Hard rules:
- The only allowed scene helper import is exactly: from helper import *
- Only use APIs that helper.py exposes.
- Scene coordinates are +x forward, +y left, +z up. The ground plane is z=0.
- Always define @register() def root_scene() -> Shape.
- Call extra registered scene functions via library_call("function_name", ...).
- Never put @register() on helpers that do not return Shape.
- Never use undefined type names like Scene, Object, Pose, Vector3, or List[Scene]. If unsure, omit the annotation.
- Never call .add(...) on keywords, Shape, or any Python list. keywords must stay Python lists; combine shapes with concat_shapes(...).
- keywords=... must be a list literal or a list variable, not a set-like object.
- Use get_object_info(...)[\"max\"][2], get_object_info(...)[\"min\"][2], and compute_shape_center(...) for placement math.
- If you rotate an object after creating it, rotate around compute_shape_center(shape), then translate or re-check height.
- Do not output Markdown fences or prose when calling generate_scene.

Copy this skeleton and replace asset ids, keywords, positions, and support relationships:
{SCENE_TEMPLATE}

Registered helper pattern:
{REGISTERED_HELPER_EXAMPLE}
""".strip()

model = OpenAIChatModel(
    os.environ.get("COPILOTKIT_MODEL", "gpt-4o"),
    provider=OpenAIProvider(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", ""),
    ),
)

agent = Agent(model, instructions=GENIE_SIM_AGENT_INSTRUCTIONS)
