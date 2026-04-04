"use client";

import { useCopilotAdditionalInstructions, useCopilotReadable, useFrontendTool } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-core/v2"
import { useEffect, useState } from "react"
import {
  SandpackFiles,
  SandpackPreview,
  SandpackProvider,
  SandpackState,
  defaultLight,
  useSandpack
} from "@codesandbox/sandpack-react"
import { CustomProvider } from "@mariozechner/pi-web-ui";
import Pi from "../components/agent/pi";

const SETTINGS = {
    'proxy.enabled': true,
}

const baseUrl = Object.assign(new URL(location.href), {
     pathname: '/cors/moonshot/v1'
}).toString()

const PROVIDER = {
    id: 'moonshot',
    name: 'moonshot',
    baseUrl,
    type: 'openai-completions',
    models: [{
        id: 'kimi-k2.5',
        name: 'Kimi K2.5',
        api: 'openai-completions',
        provider: 'moonshot',
        baseUrl,
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

const APP_CODE = `
import { useState, useEffect } from 'react'
import Entry from './entry'
import Props from './props.json'
export default function App() {
    const [props, setProps] = useState(Props)
    useEffect(() => {
        function onMessage(evt) {
            console.log(evt.data)
            if (evt.data.props) {
                setProps(evt.data.props)
            }
        }
        window.addEventListener('message', onMessage)
        return () => {
            window.removeEventListener('message', onMessage)
        }
    }, [])
    return <Entry { ...props } />
}
`

const DEFAULT_FILES = {
    '/App.js': `export default () => "Hi"`
} as SandpackFiles

function FetchSandpack({ setSandpack }: { setSandpack: (value: SandpackState) => void }) {
    const { sandpack } = useSandpack()
    useEffect(() => setSandpack(sandpack), [Object.keys(sandpack.clients).join(';'), sandpack.error])
    return null
}

export default function HomePage() {
    useCopilotAdditionalInstructions({
        instructions: `
            You are a professional React developer who can develop web-based applications.
            Just output front end code and no npm commands are required.
        `
    }, [])

    const [props, setProps] = useState({ })
    useCopilotReadable({
        description: "Current entry component properties that the assistant can revise or keep stable while changing props.",
        value: props
    }, [props])

    const [files, setFiles] = useState(DEFAULT_FILES)
    useFrontendTool({
        name: "set_app_code",
        description:
            "Create or replace the live React preview. `entry` must default export a React component. `defaultProps` must be a JSON object.",
        followUp: true,
        parameters: [{
            name: "entry",
            type: "string",
            description: "React component source code that default exports a component.",
            required: true
        }, {
            name: "props",
            type: "object",
            description: "JSON props passed into the generated component.",
            required: true
        }, {
            name: "files",
            type: "object",
            description: "Additional files with file path as key, file content as value",
            required: true
        }],
        handler: ({ entry, props, files }) => {
            setProps(props)
            setFiles({
                ...files,
                '/App.js': APP_CODE,
                '/entry.tsx': entry,
                '/props.json': JSON.stringify(props),
            })
            return { ok: true };
        },
        render: ({ status }) => {
            if (status === "inProgress") {
                return <div className="tool-badge">Building preview...</div>;
            }
            return <div className="tool-badge">Preview updated</div>;
        }
    }, []);

    const [sandpack, setSandpack] = useState(undefined as undefined | SandpackState)
    useFrontendTool({
        name: "set_app_props",
        description:
            "Update only the JSON props for the current preview without replacing the component code. Use this to iterate with the user after a preview already exists.",
        followUp: true,
        parameters: [{
            name: "props",
            type: "object",
            description: "The full JSON props object that should be passed to the current preview component.",
            required: true
        }],
        handler: ({ props }) => {
            setProps(props)
            for (const client of Object.values(sandpack?.clients || { })) {
                client.iframe.contentWindow?.postMessage({ props }, "*")
            }
            return { ok: true }
        },
        render: ({ status }) => {
            if (status === "inProgress") {
                return <div className="tool-badge">Updating props...</div>;
            }
            return <div className="tool-badge">Props updated</div>;
        }
    }, []);

    const hasApp = '/App.js' in files,
        width = hasApp ? 400 : '100%' 
    return <div className="h-full w-full flex">
        {
            hasApp &&
            <div className="flex-1 h-full">
                <SandpackProvider
                    style={{ height: '100%' }}
                    template="react"
                    theme={defaultLight}
                    files={files}
                    customSetup={{
                        npmRegistries: [{
                            registryUrl: '/npm',
                            enabledScopes: [],
                            limitToScopes: false,
                            proxyEnabled: false,
                        }]
                    }}
                    options={{
                        bundlerURL: 'http://dev.yff.me:13000/',
                        recompileMode: 'delayed',
                        recompileDelay: 250
                    }}>
                    <FetchSandpack setSandpack={ setSandpack } />
                    <SandpackPreview
                        className="h-full"
                        showOpenInCodeSandbox={false}
                        showOpenNewtab={false}
                        showSandpackErrorOverlay={true}
                    />
                </SandpackProvider>
            </div>
        }
        {
            0 ?
            <CopilotChat className="copilotkit-fix" style={{ width }} /> :
            <Pi provider={ PROVIDER } settings={ SETTINGS } style={{ width }} />
        }
    </div>
}
