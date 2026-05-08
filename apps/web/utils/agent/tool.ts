import { useFrontendTool } from "@copilotkit/react-core";
import { z } from "zod";

export const useTool = ((args: any, deps: any) => {
    const { handler } = args
    return useFrontendTool({
        ...args,
        async handler(args: any) {
            try {
                return await handler(args)
            } catch (error) {
                return { error: formatToolError(error) }
            }
        }
    }, deps)
}) as typeof useFrontendTool

export function formatToolError(error: unknown) {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return String(error);
}

type ToolParameters = NonNullable<Parameters<typeof useFrontendTool>[0]["parameters"]>;
type ToolParameter = ToolParameters[number];
type MergedToolParameter = {
  parameter: ToolParameter;
  presentIn: number;
  requiredInAllVariants: boolean;
};
type TypedToolArgs<TSchema extends z.AnyZodObject> = {
  available?: "disabled" | "enabled";
  description?: string;
  followUp?: boolean;
  handler?: (args: z.infer<TSchema>) => Promise<unknown> | unknown;
  name: string;
  parameters: TSchema;
  render?: Parameters<typeof useFrontendTool>[0]["render"];
};

export function useTypedTool<TSchema extends z.AnyZodObject>(
  args: TypedToolArgs<TSchema>,
  deps?: ReadonlyArray<unknown>,
) {
  const { handler, parameters } = args;
  useTool({
    ...args,
    parameters: buildToolParametersFromZod(parameters),
    handler: handler ? async (toolArgs: unknown) => await handler(parameters.parse(toolArgs)) : undefined,
  }, deps as any[]);
}

export function buildToolParametersFromZod(schema: z.AnyZodObject): ToolParameters {
  const shape = schema.shape as z.ZodRawShape;
  return Object.entries(shape).map(([name, field]) => buildToolParameter(name, field));
}

function buildToolParameter(name: string, field: z.ZodTypeAny): ToolParameter {
  const required = !(field instanceof z.ZodOptional || field instanceof z.ZodDefault);
  const schema = unwrapSchema(field);
  const description = field.description ?? schema.description;

  if (schema instanceof z.ZodEnum) return { name, type: "string", required, description, enum: [...schema.options] };
  if (schema instanceof z.ZodLiteral && typeof schema._def.value === "string") return { name, type: "string", required, description, enum: [schema._def.value] };
  if (schema instanceof z.ZodString) return { name, type: "string", required, description };
  if (schema instanceof z.ZodNumber) return { name, type: "number", required, description };
  if (schema instanceof z.ZodBoolean) return { name, type: "boolean", required, description };
  if (schema instanceof z.ZodArray) return { name, type: getArrayParameterType(name, schema.element), required, description };
  if (schema instanceof z.ZodObject) return { name, type: "object", required, description, attributes: buildToolParametersFromZod(schema) };
  if (schema instanceof z.ZodDiscriminatedUnion) return { name, type: "object", required, description, attributes: mergeObjectVariants([...schema.options]) };
  if (schema instanceof z.ZodTuple) return { name, type: `${getPrimitiveArrayItemType(name, schema.items)}[]`, required, description };
  throw new Error(`Unsupported zod schema for tool parameter "${name}"`);
}

function mergeObjectVariants(schemas: readonly z.ZodTypeAny[]): ToolParameters {
  const merged = new Map<string, MergedToolParameter>();
  for (const schema of schemas) {
    const objectSchema = unwrapSchema(schema);
    if (!(objectSchema instanceof z.ZodObject)) throw new Error("Expected an object schema");
    for (const parameter of buildToolParametersFromZod(objectSchema)) {
      const current = merged.get(parameter.name);
      if (!current) {
        merged.set(parameter.name, {
          parameter,
          presentIn: 1,
          requiredInAllVariants: parameter.required !== false,
        });
        continue;
      }
      merged.set(parameter.name, {
        parameter: mergeToolParameters(current.parameter, parameter),
        presentIn: current.presentIn + 1,
        requiredInAllVariants: current.requiredInAllVariants && parameter.required !== false,
      });
    }
  }
  return [...merged.values()].map(({ parameter, presentIn, requiredInAllVariants }) => ({
    ...parameter,
    required: presentIn === schemas.length && requiredInAllVariants,
  }));
}

function mergeToolParameters(current: ToolParameter, next: ToolParameter): ToolParameter {
  if (current.type === "string" && next.type === "string") {
    const enumValues = new Set([...(current.enum ?? []), ...(next.enum ?? [])]);
    return {
      ...current,
      ...(enumValues.size > 0 ? { enum: [...enumValues] } : {}),
    };
  }
  return current;
}

function getPrimitiveArrayItemType(name: string, items: readonly z.ZodTypeAny[]) {
  const first = unwrapSchema(items[0]!);
  if (!(first instanceof z.ZodString || first instanceof z.ZodNumber || first instanceof z.ZodBoolean)) {
    throw new Error(`Unsupported tuple item schema for tool parameter "${name}"`);
  }
  for (const item of items.slice(1)) {
    const current = unwrapSchema(item);
    if (current.constructor !== first.constructor) throw new Error(`Mixed tuple item schema for tool parameter "${name}"`);
  }
  if (first instanceof z.ZodString) return "string";
  if (first instanceof z.ZodNumber) return "number";
  return "boolean";
}

function getArrayParameterType(name: string, element: z.ZodTypeAny) {
  const item = unwrapSchema(element);
  if (item instanceof z.ZodString || item instanceof z.ZodEnum || (item instanceof z.ZodLiteral && typeof item._def.value === "string")) {
    return "string[]";
  }
  if (item instanceof z.ZodNumber) return "number[]";
  if (item instanceof z.ZodBoolean) return "boolean[]";
  if (item instanceof z.ZodObject || item instanceof z.ZodDiscriminatedUnion) return "object[]";
  throw new Error(`Unsupported array item schema for tool parameter "${name}"`);
}

function unwrapSchema(schema: z.ZodTypeAny): z.ZodTypeAny {
  let current = schema;
  while (current instanceof z.ZodDefault || current instanceof z.ZodNullable || current instanceof z.ZodOptional) {
    current = current._def.innerType;
  }
  return current;
}
