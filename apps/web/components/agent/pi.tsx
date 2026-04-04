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
  CustomProvider,
  ModelSelector,
} from '@mariozechner/pi-web-ui';

import './app.css';
import { DetailedHTMLProps, HTMLAttributes, useEffect, useRef } from 'react';

function setupStorage(options: { provider?: CustomProvider, settings?: Record<string, any> }) {
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
    }

    const storage = new AppStorage(settings, providerKeys, sessions, customProvider, backend);
    setAppStorage(storage);

    return storage
}

type DivProps = DetailedHTMLProps<HTMLAttributes<HTMLDivElement>, HTMLDivElement>

export default function Pi(props: DivProps & {
    settings?: Record<string, any>
    provider?: CustomProvider
    systemPrompt?: string
    thinkingLevel?: ThinkingLevel
}) {
    const div = useRef<null | HTMLDivElement>(null)
    useEffect(() => {
        if (!div.current) {
            return () => { }
        }

        const { settings, provider, systemPrompt, thinkingLevel } = props
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
        chatPanel.setAgent(agent, {
            onModelSelect: () => ModelSelector.open(agent.state.model, model => agent.setModel(model), ['moonshot']),
            onApiKeyRequired: provider => ApiKeyPromptDialog.prompt(provider),
        })

        div.current.appendChild(chatPanel)
        return () => {
            div.current?.removeChild(chatPanel)
            agent.abort()
        }
    }, [])
    return <div ref={ div } { ...props }></div>
}
