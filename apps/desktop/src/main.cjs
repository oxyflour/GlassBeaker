// @ts-check
const path = require("node:path"),
    { spawn } = require('child_process'),
    { app, BrowserWindow, utilityProcess } = require("electron/main"),
    { existsSync, readFileSync } = require('fs'),
    env = existsSync('.env') ? readFileSync('.env', 'utf8') : ''

for (const line of env.split('\n')) {
    const [key, val] = line.trim().split('=').map(item => item.trim())
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
function resolvePythonRuntime(label = '') {
    if (!app.isPackaged) {
        return label === 'ros' ? {
            label,
            command: 'pixi',
            args: ['run', '--no-install', 'python', '-u', 'app.py'],
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

async function startServer(nextJsPort = 13000, pythonPort = 13001) {
    const pyRuntime = resolvePythonRuntime('python')
    watchProc(pyRuntime.label, spawn(pyRuntime.command, pyRuntime.args, {
        env: { ...process.env, LISTEN_PORT: `${pythonPort}`, NO_PROXY: '*' },
        cwd: pyRuntime.cwd,
        stdio: 'pipe'
    }))

    const rosRuntime = resolvePythonRuntime('ros')
    watchProc(rosRuntime.label, spawn(rosRuntime.command, rosRuntime.args, {
        env: { ...process.env, WS_ADDR: `ws://127.0.0.1:${pythonPort}/api/ros/ws`, NO_PROXY: '*' },
        cwd: rosRuntime.cwd,
        stdio: 'pipe'
    }))

    const apiRuntime = await assertUrl(`http://127.0.0.1:${pythonPort}/runtime`)
    console.log(`[main] RUNTIME: ${apiRuntime}`)

    const nextjs = utilityProcess.fork(path.join(root, 'web/node_modules/next/dist/bin/next'), [
        '-p', `${nextJsPort}`
    ], {
        env: { ...process.env, API_REWRITE: `http://127.0.0.1:${pythonPort}/`, API_RUNTIME: apiRuntime },
        cwd: path.join(root, 'web'),
        stdio: "pipe"
    });
    // @ts-ignore
    watchProc('nextjs', nextjs)

    const url = `http://localhost:${nextJsPort}`
    await assertUrl(url)
    return url
}

async function createMainWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 760,
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
