// @ts-check
const path = require("node:path"),
    { spawn } = require('child_process'),
    { app, BrowserWindow, utilityProcess } = require("electron/main"),
    { existsSync, mkdirSync, readFileSync, writeFileSync } = require('fs'),
    env = existsSync('.env') ? readFileSync('.env', 'utf8') : ''

for (const line of env.split('\n')) {
    const separator = line.indexOf('='),
        key = separator >= 0 ? line.slice(0, separator).trim() : '',
        rawVal = separator >= 0 ? line.slice(separator + 1).trim() : '',
        quote = rawVal[0],
        val = (quote === '"' || quote === "'") && rawVal.endsWith(quote)
            ? rawVal.slice(1, -1)
            : rawVal
    if (key && !key.startsWith('#')) {
        console.log(`[main] updated env ${key}`)
        process.env[key] = val
    }
}

/**
 * 
 * @param { string } label 
 * @param { string } data 
 */
function logWithLabel(label, data) {
    for (const line of `${data}`.split('\n')) {
        line && console.log(`[${label}] ${line}`)
    }
}

/**
 * 
 * @param { string } label 
 * @param { import('child_process').ChildProcess } proc 
 */
function watchProc(label, proc) {
    proc.stdout?.on('data', data => logWithLabel(label, data))
    proc.stderr?.on('data', data => logWithLabel(label, data))
    proc.addListener('error', error => {
        console.error(`[main] ERR: ${label} failed`, error)
    })
    proc.addListener('exit', (code, signal) => {
        console.log(`[main] BYE: ${label} quit (code=${code}, signal=${signal})`)
        app.quit()
    })
}

/**
 * @type { null | import('electron').BrowserWindow }
 */
let mainWindow = null;

/**
 * 
 * @param { string } url 
 */
async function assertUrl(url, retry = 30){
    while (retry -- > 0) {
        await new Promise(resolve => setTimeout(resolve, 1000))
        try {
            const req = await fetch(url),
                text = await req.text()
            if (req.status === 200) {
                return text
            } else {
                throw Error(`${req.status}: ${text}`)
            }
        } catch (err) {
            console.warn(`[main] waiting for url ${url} (${retry} retries left)`)
        }
    }
    throw Error(`failed to request ${url}`)
}

const root = app.isPackaged ? process.resourcesPath : path.resolve(__dirname, "..", "..")
const messagingEnvPrefixes = [
    "TELEGRAM_",
    "DISCORD_",
    "WHATSAPP_",
    "SLACK_",
    "SIGNAL_",
    "EMAIL_",
    "SMS_",
    "MATTERMOST_",
    "MATRIX_",
    "DINGTALK_",
    "FEISHU_",
    "WECOM_",
    "WEIXIN_",
    "BLUEBUBBLES_",
    "QQ_",
    "YUANBAO_",
    "WEBHOOK_",
    "MSGRAPH_WEBHOOK_",
    "HOMEASSISTANT_",
]

function yamlString(value) {
    return JSON.stringify(`${value || ''}`)
}

function dotenvLine(key, value) {
    const sanitized = `${value || ''}`.replace(/\r?\n/g, '')
        .replace(/\\/g, '\\\\')
        .replace(/"/g, '\\"')
    return `${key}="${sanitized}"`
}

function syncHermesProfile(hermesHome) {
    const openaiBaseUrl = (process.env.OPENAI_BASE_URL || '').trim(),
        model = (process.env.COPILOTKIT_MODEL || process.env.OPENAI_MODEL || '').trim()

    if (!openaiBaseUrl || !model) {
        return false
    }

    mkdirSync(hermesHome, { recursive: true })
    writeFileSync(path.join(hermesHome, "config.yaml"), [
        "model:",
        "  provider: custom",
        `  default: ${yamlString(model)}`,
        `  base_url: ${yamlString(openaiBaseUrl)}`,
        "  api_mode: chat_completions",
        "",
    ].join('\n'), 'utf8')

    writeFileSync(path.join(hermesHome, ".env"), [
        ["OPENAI_API_KEY", process.env.OPENAI_API_KEY],
        ["OPENAI_BASE_URL", openaiBaseUrl],
        ["COPILOTKIT_MODEL", model],
    ].filter(([, value]) => `${value || ''}`.trim())
        .map(([key, value]) => dotenvLine(key, value))
        .join('\n') + '\n', 'utf8')

    console.log(`[main] synced Hermes profile at ${hermesHome}`)
    return true
}

function resolvePythonRuntime(label = '') {
    if (!app.isPackaged) {
        return label === 'ros' ? {
            label,
            command: 'pixi',
            args: ['run', '--no-install', 'python', '-u', 'app.py'],
            cwd: path.join(root, label)
        } : label === 'hermes' ? {
            label,
            command: 'uv',
            args: ['run', '--no-sync', 'python', '-u', 'app.py'],
            cwd: path.join(root, label)
        } : {
            label,
            command: 'uv',
            args: ['run', '--no-sync', 'python', '-u', 'app.py'],
            cwd: path.join(root, label)
        }
    }

    const exeName = process.platform === 'win32'
            ? 'glassbeaker-python.exe'
            : 'glassbeaker-python',
        cwd = path.join(root, 'python', 'glassbeaker-python'),
        command = path.join(cwd, exeName)

    if (!existsSync(command)) {
        throw Error(`missing packaged python executable: ${command}`)
    }

    return { label, command, args: [], cwd }
}

/**
 * 
 * @param { string } label 
 * @param { Record<string, string> } env 
 * @param { import('child_process').SpawnOptions } opts 
 */
function startPythonModule(label, env, opts = {}) {
    const pyRuntime = resolvePythonRuntime(label),
        { env: baseEnv = process.env, ...spawnOpts } = opts
    watchProc(pyRuntime.label, spawn(pyRuntime.command, pyRuntime.args, {
        env: { ...baseEnv, ...env },
        cwd: pyRuntime.cwd,
        stdio: 'pipe',
        ...spawnOpts,
    }))
    return pyRuntime
}

async function startServer(nextJsPort = 13000, pythonPort = 13001) {
    const isaacRuntime = `http://127.0.0.1:${nextJsPort}/api/isaac`,
        hermesHome = process.env.GLASSBEAKER_HERMES_HOME
            || path.join(app.getPath("home"), ".glass-beaker", "hermes"),
        hermesPort = process.env.GLASSBEAKER_HERMES_PORT || '13002',
        hermesEnv = { ...process.env }

    const hermesUsesCustomEndpoint = syncHermesProfile(hermesHome)

    for (const key of Object.keys(hermesEnv)) {
        if (messagingEnvPrefixes.some(prefix => key.startsWith(prefix))) {
            delete hermesEnv[key]
        }
    }

    startPythonModule('python', {
        NO_PROXY: '*',
        LISTEN_PORT: `${pythonPort}`,
        ISAAC_API_URL: isaacRuntime,
    }) 
    startPythonModule('ros', {
        NO_PROXY: '*',
        WS_ADDR: `ws://127.0.0.1:${pythonPort}/api/ros/ws`,
    })
    startPythonModule('hermes', {
        NO_PROXY: '*',
        HERMES_HOME: hermesHome,
        ...(hermesUsesCustomEndpoint ? { HERMES_INFERENCE_PROVIDER: 'custom' } : {}),
        API_SERVER_ENABLED: 'true',
        API_SERVER_HOST: '127.0.0.1',
        API_SERVER_PORT: `${hermesPort}`,
        API_SERVER_KEY: 'sk-1234',
        API_SERVER_CORS_ORIGINS: `http://localhost:${nextJsPort},http://127.0.0.1:${nextJsPort}`,
        GATEWAY_ALLOW_ALL_USERS: 'true',
    }, { env: hermesEnv })

    const url = `http://localhost:${nextJsPort}`,
        nextjs = utilityProcess.fork(path.join(root, 'web/node_modules/next/dist/bin/next'), [
        '-p', `${nextJsPort}`
    ], {
        env: {
            ...process.env,
            API_REWRITE: `http://127.0.0.1:${pythonPort}/`,
            GLASSBEAKER_HERMES_PORT: `${hermesPort}`,
        },
        cwd: path.join(root, 'web'),
        stdio: "pipe"
    })
    // @ts-ignore
    watchProc('nextjs', nextjs)

    const [apiRuntime] = await Promise.all([
        assertUrl(`http://127.0.0.1:${pythonPort}/runtime`),
        assertUrl(url),
    ])
    console.log(`[main] RUNTIME: ${apiRuntime}`)
    return url
}

async function createMainWindow() {
    mainWindow = new BrowserWindow({
        width: 640,
        height: 480,
        show: false,
        webPreferences: {
            additionalArguments: [`--glassbeaker-packaged=${app.isPackaged ? "1" : "0"}`],
            preload: path.join(__dirname, "preload.cjs")
        }
    });

    mainWindow.once("ready-to-show", () => {
        mainWindow?.show();
    });

    mainWindow.on("closed", () => {
        mainWindow = null;
    });

    await mainWindow.loadFile(path.join(root, 'desktop', 'index.html'))

    const url = await startServer();
    await mainWindow.loadURL(url);
}

app.on("before-quit", () => {
    // cleanup
});

app.whenReady().then(async () => {
    try {
        await createMainWindow();
    } catch (error) {
        console.error("Failed to start desktop app:", error);
        app.quit();
    }
});

app.on("window-all-closed", () => {
    if (process.platform !== "darwin") {
        app.quit();
    }
});

app.on("activate", async () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        await createMainWindow();
    }
});
