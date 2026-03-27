const http = require("node:http");
const path = require("node:path");
const { app, BrowserWindow, utilityProcess } = require("electron");

const HOST = "127.0.0.1";
const PORT = process.env.GLASSBEAKER_PORT || "3000";
const PYTHON_HOST = process.env.GLASSBEAKER_PYTHON_HOST || HOST;
const PYTHON_PORT = process.env.GLASSBEAKER_PYTHON_PORT || "8000";

let mainWindow = null;
let serverProcess = null;
let isQuitting = false;

function pipeUtilityLogs(stream, label) {
  if (!stream) {
    return;
  }

  stream.on("data", (chunk) => {
    process.stdout.write(`[${label}] ${chunk}`);
  });
}

function pingServer(url) {
  return new Promise((resolve, reject) => {
    const request = http.get(url, (response) => {
      response.resume();

      if (response.statusCode && response.statusCode < 500) {
        resolve();
        return;
      }

      reject(new Error(`Unexpected status code: ${response.statusCode}`));
    });

    request.on("error", reject);
    request.setTimeout(1500, () => {
      request.destroy(new Error("Timed out"));
    });
  });
}

async function waitForServer(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;

  while (Date.now() < deadline) {
    try {
      await pingServer(url);
      return;
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }

  throw lastError || new Error(`Failed to start server at ${url}`);
}

function getWebRoot() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "web");
  }

  return path.resolve(__dirname, "..", "..", "web");
}

function getPythonRoot() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "python");
  }

  return path.resolve(__dirname, "..", "..", "python");
}

function startServer() {
  if (serverProcess) {
    return;
  }

  const childPath = path.join(__dirname, "server-process.cjs");

  serverProcess = utilityProcess.fork(childPath, [], {
    env: {
      ...process.env,
      ELECTRON_SERVER_MODE: app.isPackaged ? "production" : "development",
      GLASSBEAKER_HOST: HOST,
      GLASSBEAKER_PORT: PORT,
      GLASSBEAKER_WEB_DIR: getWebRoot(),
      GLASSBEAKER_PYTHON_HOST: PYTHON_HOST,
      GLASSBEAKER_PYTHON_PORT: PYTHON_PORT,
      GLASSBEAKER_PYTHON_ROOT: getPythonRoot()
    },
    stdio: "pipe"
  });

  pipeUtilityLogs(serverProcess.stdout, "server");
  pipeUtilityLogs(serverProcess.stderr, "server");

  serverProcess.on("exit", (code) => {
    if (!isQuitting) {
      console.error(`utility process exited with code ${code}`);
      app.quit();
    }

    serverProcess = null;
  });
}

async function createMainWindow() {
  const url = `http://${HOST}:${PORT}`;
  const pythonHealthUrl = `http://${PYTHON_HOST}:${PYTHON_PORT}/healthz`;
  startServer();
  await Promise.all([
    waitForServer(url, app.isPackaged ? 30000 : 90000),
    waitForServer(pythonHealthUrl, app.isPackaged ? 30000 : 90000)
  ]);

  mainWindow = new BrowserWindow({
    width: 1200,
    height: 760,
    minWidth: 960,
    minHeight: 640,
    backgroundColor: "#07111f",
    show: false,
    webPreferences: {
      additionalArguments: [`--glassbeaker-packaged=${app.isPackaged ? "1" : "0"}`],
      preload: path.join(__dirname, "preload.cjs")
    }
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  await mainWindow.loadURL(url);
}

function stopServer() {
  if (!serverProcess) {
    return;
  }

  serverProcess.kill();
  serverProcess = null;
}

app.on("before-quit", () => {
  isQuitting = true;
  stopServer();
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
