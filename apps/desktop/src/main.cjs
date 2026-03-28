// @ts-check
const path = require("node:path"),
    { spawn } = require('child_process'),
    { app, BrowserWindow, utilityProcess } = require("electron")

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
    proc.addListener('exit', () => {
        console.log(`BYE: ${label} quit`)
        app.quit()
    })
}

/**
 * @type { null | Electron.BrowserWindow }
 */
let mainWindow = null;

const root = app.isPackaged ? process.resourcesPath : path.resolve(__dirname, "..", "..")
async function startServer() {
    const next = utilityProcess.fork(path.join(root, 'web/node_modules/next/dist/bin/next'), [], {
        cwd: path.join(root, 'web'),
        stdio: "pipe"
    });
    // @ts-ignore
    watchProc('nextjs', next)

    const python = spawn('uv', ['run', 'app.py'], {
        shell: true,
        cwd: path.join(root, 'python'),
        stdio: 'pipe'
    })
    watchProc('python', python)

    const url = 'http://localhost:3000'
    while (true) {
        await new Promise(resolve => setTimeout(resolve, 1000))
        try {
            await fetch(url)
            break
        } catch (err) {
            console.warn(`waiting for url ${url}`)
        }
    }
    return url
}

async function createMainWindow() {
    const url = await startServer();
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 760,
        backgroundColor: "#07111f",
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
