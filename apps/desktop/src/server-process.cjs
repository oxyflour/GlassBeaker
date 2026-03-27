const path = require("node:path");
const { spawn } = require("node:child_process");

const mode = process.env.ELECTRON_SERVER_MODE;
const host = process.env.GLASSBEAKER_HOST || "127.0.0.1";
const port = process.env.GLASSBEAKER_PORT || "3000";
const webRoot = process.env.GLASSBEAKER_WEB_DIR;
const pythonHost = process.env.GLASSBEAKER_PYTHON_HOST || host;
const pythonPort = process.env.GLASSBEAKER_PYTHON_PORT || "8000";
const pythonRoot = process.env.GLASSBEAKER_PYTHON_ROOT;
const pythonVersion = process.env.GLASSBEAKER_PYTHON_VERSION || "3.12";

const childProcesses = new Set();
let isShuttingDown = false;

function getNodeCommand() {
  return process.platform === "win32" ? "node.exe" : "node";
}

function getUvCommand() {
  return process.platform === "win32" ? "uv.exe" : "uv";
}

function getPythonExecutableName() {
  return process.platform === "win32"
    ? "glassbeaker-python.exe"
    : "glassbeaker-python";
}

function trackChild(label, child) {
  childProcesses.add(child);

  child.on("error", (error) => {
    if (isShuttingDown) {
      return;
    }

    console.error(`Failed to start ${label}:`, error);
    shutdown(1);
  });

  child.on("exit", (code) => {
    childProcesses.delete(child);

    if (isShuttingDown) {
      return;
    }

    console.error(`${label} exited with code ${code}`);
    shutdown(code ?? 1);
  });

  return child;
}

function shutdown(code = 0) {
  if (isShuttingDown) {
    return;
  }

  isShuttingDown = true;

  for (const child of childProcesses) {
    child.kill();
  }

  childProcesses.clear();
  process.exit(code);
}

process.on("SIGTERM", () => shutdown(0));
process.on("SIGINT", () => shutdown(0));

function startPythonDevelopmentServer() {
  if (!pythonRoot) {
    throw new Error("GLASSBEAKER_PYTHON_ROOT is required in development mode.");
  }

  trackChild(
    "python server",
    spawn(
      getUvCommand(),
      ["run", "--project", pythonRoot, "--python", pythonVersion, "python", "app.py"],
      {
        cwd: pythonRoot,
        env: {
          ...process.env,
          GLASSBEAKER_PYTHON_HOST: pythonHost,
          GLASSBEAKER_PYTHON_PORT: pythonPort
        },
        stdio: "inherit"
      }
    )
  );
}

function startPythonPackagedServer() {
  if (!pythonRoot) {
    throw new Error("GLASSBEAKER_PYTHON_ROOT is required in production mode.");
  }

  const executable = path.join(pythonRoot, getPythonExecutableName());

  trackChild(
    "python server",
    spawn(executable, [], {
      cwd: pythonRoot,
      env: {
        ...process.env,
        GLASSBEAKER_PYTHON_HOST: pythonHost,
        GLASSBEAKER_PYTHON_PORT: pythonPort
      },
      stdio: "inherit"
    })
  );
}

if (mode === "development") {
  const nextCli = path.join(webRoot, "node_modules", "next", "dist", "bin", "next");

  startPythonDevelopmentServer();

  trackChild(
    "next dev server",
    spawn(getNodeCommand(), [nextCli, "dev", "-H", host, "-p", port], {
      cwd: webRoot,
      env: {
        ...process.env,
        NODE_ENV: "development",
        GLASSBEAKER_PYTHON_HOST: pythonHost,
        GLASSBEAKER_PYTHON_PORT: pythonPort
      },
      stdio: "inherit"
    })
  );
} else {
  startPythonPackagedServer();

  process.env.NODE_ENV = "production";
  process.env.HOSTNAME = host;
  process.env.PORT = port;
  process.env.GLASSBEAKER_PYTHON_HOST = pythonHost;
  process.env.GLASSBEAKER_PYTHON_PORT = pythonPort;

  const serverEntry = path.join(webRoot, "apps", "web", "server.js");
  process.chdir(path.dirname(serverEntry));
  require(serverEntry);
}
