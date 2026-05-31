import { Agent, ThinkingLevel } from '@mariozechner/pi-agent-core';
import {
  ChatPanel,
  AppStorage,
  IndexedDBStorageBackend,
  ProviderKeysStore,
  SessionsStore,
  SettingsStore,
  setAppStorage,
  defaultConvertToLlm,
  ApiKeyPromptDialog,
  CustomProvidersStore,
  ModelSelector,
} from '@mariozechner/pi-web-ui';

import './app.css';
import { DetailedHTMLProps, HTMLAttributes, useEffect, useRef } from 'react';
import { PROVIDER, type GlassCustomProvider } from './pi-provider';

function setupStorage(options: { provider?: GlassCustomProvider, settings?: Record<string, any> }) {
    const settings = new SettingsStore();
    const providerKeys = new ProviderKeysStore();
    const sessions = new SessionsStore();
    const customProvider = new CustomProvidersStore();

    const backend = new IndexedDBStorageBackend({
        dbName: 'glass-beaker-pi',
        version: 1,
        stores: [
            settings.getConfig(),
            providerKeys.getConfig(),
            sessions.getConfig(),
            customProvider.getConfig(),
            SessionsStore.getMetadataConfig(),
        ],
    });

    settings.setBackend(backend);
    providerKeys.setBackend(backend);
    sessions.setBackend(backend);
    customProvider.setBackend(backend);

    if (options.settings) {
        for (const key in options.settings) {
            settings.set(key, options.settings[key])
        }
    }

    if (options.provider) {
        customProvider.set(options.provider)
        for (const [providerId, apiKey] of Object.entries(options.provider.apiKeys || {})) {
            providerKeys.set(providerId, apiKey)
        }
    }

    const storage = new AppStorage(settings, providerKeys, sessions, customProvider, backend);
    setAppStorage(storage);

    return storage
}

type DivProps = DetailedHTMLProps<HTMLAttributes<HTMLDivElement>, HTMLDivElement>
type PiWebProps = DivProps & {
    settings?: Record<string, any>
    provider?: GlassCustomProvider
    systemPrompt?: string
    thinkingLevel?: ThinkingLevel
    params?: unknown
    searchParams?: unknown
}

export default function PiWeb(props: PiWebProps) {
    const {
        className,
        params: _params,
        provider = PROVIDER,
        searchParams: _searchParams,
        settings,
        style,
        systemPrompt,
        thinkingLevel,
        ...divProps
    } = props
    const div = useRef<null | HTMLDivElement>(null)
    useEffect(() => {
        if (!div.current) {
            return () => { }
        }

        setupStorage({ settings, provider })
        const agent = new Agent({
            initialState: {
                systemPrompt: systemPrompt || 'You are a helpful assistant.',
                model: provider?.models?.[0],
                thinkingLevel: thinkingLevel,
                messages: [],
                tools: [],
            },
            convertToLlm: defaultConvertToLlm,
        });

        const chatPanel = new ChatPanel()
        const allowedProviders = provider?.models?.length
            ? Array.from(new Set(provider.models.map(model => model.provider)))
            : provider ? [provider.id] : undefined
        chatPanel.setAgent(agent, {
            onModelSelect: () => ModelSelector.open(agent.state.model, model => agent.setModel(model), allowedProviders),
            onApiKeyRequired: provider => ApiKeyPromptDialog.prompt(provider),
        })

        div.current.appendChild(chatPanel)
        return () => {
            div.current?.removeChild(chatPanel)
            agent.abort()
        }
    }, [])
    return (
        <div
            ref={ div }
            className={ [
                'fixed inset-x-0 bottom-0 flex h-dvh min-h-0 flex-col overflow-hidden bg-background text-foreground',
                className,
            ].filter(Boolean).join(' ') }
            style={ style }
            { ...divProps }
        ></div>
    )
}
