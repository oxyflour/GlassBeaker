import type { useFrontendTool } from "@copilotkit/react-core";

type ToolParameter = {
  description: string;
  name: string;
  required: boolean;
  type: "array" | "boolean" | "number" | "object" | "string";
};

const DSL_REFERENCE = [
  "You are a robotics simulation scene designer. You generate scenes using the Genie Sim (AgiBotTech) scene language DSL.",
  "The DSL uses these functions to build 3D scenes from geometric primitives:",
  "",
  "## Core Functions",
  "- `primitive_call(type, shape_kwargs={...}, info={...}, color=(r,g,b))` - create a primitive shape",
  "  - type='cube': shape_kwargs={'scale': (w, h, d)} - width, height, depth",
  "  - type='sphere': shape_kwargs={'radius': r}",
  "  - type='cylinder': shape_kwargs={'radius': r, 'p0': (x,y,z), 'p1': (x,y,z)} - start/end points",
  "  - info={'id': 'object_name', 'name': 'origin'}",
  "  - color=(r, g, b) where each channel is 0.0-1.0",
  "- `transform_shape(shape, matrix)` - apply a 4x4 transform to a shape",
  "- `concat_shapes(*shapes)` - combine multiple shapes into one scene",
  "- `translation_matrix((x, y, z))` - create a translation matrix",
  "- `rotation_matrix(angle, direction=(x,y,z), point=(x,y,z))` - create rotation matrix (angle in radians)",
  "- `scale_matrix(s)` - create uniform scale matrix",
  "- `identity_matrix()` - 4x4 identity matrix",
  "",
  "## Registration",
  "- `@register('description')` - register a scene function",
  "- `library_call('func_name', **kwargs)` - call a registered function",
  "",
  "## Rules",
  "- Y-axis is UP. Place objects above the ground plane.",
  "- Use realistic proportions: table ~1.0m, cup ~0.15m, book ~0.03m thick.",
  "- Always define a function decorated with `@register()` called `root_scene` that returns the final scene.",
  "- Use `math.pi` for angle constants.",
  "- Use `np.random.rand()` for random colors if needed.",
  "- Keep scenes between 3 and 20 objects.",
  "- When user asks to modify, regenerate the full code with changes.",
  "- Output ONLY the Python code, no explanations around it.",
].join("\n");

const DSL_EXAMPLE = `Example (table with a ball):
@register("table")
def table():
    top = primitive_call("cube", shape_kwargs={"scale": (1.0, 0.05, 0.6)}, info={"id": "table_top", "name": "origin"}, color=(0.6, 0.4, 0.2))
    top = transform_shape(top, translation_matrix((0, 0.5, 0)))
    leg_h = 0.5
    legs = []
    for dx, dz in [(-0.4, -0.25), (0.4, -0.25), (-0.4, 0.25), (0.4, 0.25)]:
        leg = primitive_call("cube", shape_kwargs={"scale": (0.04, leg_h, 0.04)}, info={"id": "leg", "name": "origin"}, color=(0.5, 0.3, 0.1))
        legs.append(transform_shape(leg, translation_matrix((dx, leg_h/2, dz))))
    return concat_shapes(top, *legs)

@register("ball")
def ball():
    s = primitive_call("sphere", shape_kwargs={"radius": 0.1}, info={"id": "ball", "name": "origin"}, color=(0.9, 0.2, 0.2))
    return transform_shape(s, translation_matrix((0, 0.6, 0)))

@register("root")
def root_scene():
    return concat_shapes(table(), ball())`;

export const SCENE_ADDITIONAL_INSTRUCTIONS = [
  DSL_REFERENCE,
  "",
  "Example code:",
  DSL_EXAMPLE,
].join("\n");

export const SET_SCENE_DESCRIPTION =
  "Generate a 3D scene using Genie Sim DSL Python code. The code will be executed on the backend using genie_sim's scene language runtime.";

const SCENE_PARAMETERS: ToolParameter[] = [
  {
    name: "code",
    type: "string",
    description: "Python code using genie_sim DSL (primitive_call, transform_shape, concat_shapes, @register, etc.)",
    required: true,
  },
];

type CopilotParameters = Parameters<typeof useFrontendTool>[0]["parameters"];
export const SCENE_PARAMETERS_CPK = SCENE_PARAMETERS as unknown as CopilotParameters;
