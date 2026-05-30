import { CustomProvider } from "@mariozechner/pi-web-ui"

export const PROVIDER = {
    id: 'moonshot',
    name: 'moonshot',
    baseUrl: 'http://localhost:13000/cors/moonshot/v1',
    type: 'openai-completions',
    models: [{
        id: 'hermes',
        name: 'Hermes',
        api: 'openai-completions',
        provider: 'moonshot',
        baseUrl: 'http://localhost:13000/cors/hermes/v1',
        reasoning: true,
        input: ['text'],
        contextWindow: 131072,
        maxTokens: 32000,
        cost: {
            input: 0,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
        }
    }, {
        id: 'kimi-k2.5',
        name: 'Kimi K2.5',
        api: 'openai-completions',
        provider: 'moonshot',
        baseUrl: 'http://localhost:13000/cors/moonshot/v1',
        reasoning: false,
        input: ['text'],
        contextWindow: 131072,
        maxTokens: 32000,
        cost: {
            input: 0,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
        }
    }]
} satisfies CustomProvider
