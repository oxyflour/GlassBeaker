'use client'

import type { MutableRefObject, ReactNode } from "react";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

import type { PiFrontendToolDefinition, PiFrontendToolParameter, PiFrontendToolRequestEvent } from "./protocol";

type PiFrontendToolOptions = PiFrontendToolDefinition & {
  handler?: (args: any) => unknown | Promise<unknown>;
  render?: string | ((props: any) => ReactNode);
};

type RegisteredTool = {
  definition: PiFrontendToolDefinition;
  handler: MutableRefObject<PiFrontendToolOptions["handler"]>;
};

type ToolBucket = Map<string, RegisteredTool>;
type ToolRegistry = Map<string, ToolBucket>;

type PiFrontendToolsContextValue = {
  definitions: PiFrontendToolDefinition[];
  executeTool: (event: PiFrontendToolRequestEvent) => Promise<unknown>;
  registerTool: (name: string, registrationId: string, tool: RegisteredTool) => void;
  unregisterTool: (name: string, registrationId: string) => void;
};

const PiFrontendToolsContext = createContext<PiFrontendToolsContextValue | null>(null);

function getLatestTool(bucket: ToolBucket | undefined) {
  if (!bucket || bucket.size === 0) {
    return undefined;
  }
  return Array.from(bucket.values()).at(-1);
}

function mergeTool(registry: ToolRegistry, name: string, registrationId: string, tool?: RegisteredTool) {
  const next = new Map(registry);
  const bucket = new Map(next.get(name) || []);

  if (tool) {
    bucket.set(registrationId, tool);
  } else {
    bucket.delete(registrationId);
  }

  if (bucket.size > 0) {
    next.set(name, bucket);
  } else {
    next.delete(name);
  }
  return next;
}

export function PiFrontendToolProvider(props: { children: ReactNode }) {
  const [registry, setRegistry] = useState<ToolRegistry>(new Map());
  const registryRef = useRef(registry);
  useEffect(() => void (registryRef.current = registry), [registry]);

  const registerTool = useCallback((name: string, registrationId: string, tool: RegisteredTool) => {
    setRegistry((current) => mergeTool(current, name, registrationId, tool));
  }, []);

  const unregisterTool = useCallback((name: string, registrationId: string) => {
    setRegistry((current) => mergeTool(current, name, registrationId));
  }, []);

  const definitions = useMemo(
    () =>
      Array.from(registry.entries())
        .map(([, bucket]) => getLatestTool(bucket)?.definition)
        .filter((tool): tool is PiFrontendToolDefinition => !!tool),
    [registry],
  );

  const executeTool = useCallback(async (event: PiFrontendToolRequestEvent) => {
    const tool = getLatestTool(registryRef.current.get(event.toolName));
    if (!tool?.handler.current) {
      throw new Error(`Frontend tool "${event.toolName}" is not available.`);
    }
    return await tool.handler.current(event.args);
  }, []);

  const value = useMemo(
    () => ({ definitions, executeTool, registerTool, unregisterTool }),
    [definitions, executeTool, registerTool, unregisterTool],
  );

  return <PiFrontendToolsContext.Provider value={ value }>{ props.children }</PiFrontendToolsContext.Provider>;
}

export function usePiFrontendTool(tool: PiFrontendToolOptions, dependencies?: any[]) {
  const context = useContext(PiFrontendToolsContext);
  if (!context) {
    throw new Error("usePiFrontendTool must be used inside PiFrontendToolProvider.");
  }
  const { registerTool, unregisterTool } = context;

  const registrationId = useRef(`pi-tool-${Math.random().toString(36).slice(2)}`);
  const handlerRef = useRef<PiFrontendToolOptions["handler"]>(tool.handler);
  const definitionKey = JSON.stringify({
    available: tool.available,
    description: tool.description,
    followUp: tool.followUp,
    name: tool.name,
    parameters: tool.parameters,
  });

  useEffect(() => {
    handlerRef.current = tool.handler;
  }, [tool.handler, ...(dependencies ?? [])]);

  const definition = useMemo(
    () => ({
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters.map((parameter: PiFrontendToolParameter) => ({ ...parameter })),
      followUp: tool.followUp,
      available: tool.available,
    }),
    [definitionKey],
  );

  useEffect(() => {
    registerTool(tool.name, registrationId.current, { definition, handler: handlerRef });
    return () => unregisterTool(tool.name, registrationId.current);
  }, [definition, definitionKey, registerTool, tool.name, unregisterTool]);
}

export function usePiFrontendToolsBridge() {
  return useContext(PiFrontendToolsContext);
}
