import { useFrontendTool } from "@copilotkit/react-core";

export const useTool = ((args: any, deps: any) => {
    const { handler } = args
    return useFrontendTool({
        ...args,
        async handler(args: any) {
            try {
                return await handler(args)
            } catch (error) {
                return { error }
            }
        }
    }, deps)
}) as typeof useFrontendTool
