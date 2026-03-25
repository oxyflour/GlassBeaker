const path = require("node:path");
const { spawn } = require("node:child_process");

const mode = process.env.ELECTRON_SERVER_MODE;
const host = process.env.GLASSBEAKER_HOST || "127.0.0.1";
const port = process.env.GLASSBEAKER_PORT || "3000";
const webRoot = process.env.GLASSBEAKER_WEB_DIR;

let devServer = null;

function getNodeCommand() {
  return process.platform === "win32" ? "node.exe" : "node";
}

function shutdown(code = 0) {
  if (devServer) {
    devServer.kill();
    devServer = null;
  }

  process.exit(code);
}

process.on("SIGTERM", () => shutdown(0));
process.on("SIGINT", () => shutdown(0));

if (mode === "development") {
  const nextCli = path.join(webRoot, "node_modules", "next", "dist", "bin", "next");

  devServer = spawn(
    getNodeCommand(),
    [nextCli, "dev", "-H", host, "-p", port],
    {
      cwd: webRoot,
      env: {
        ...process.env,
        NODE_ENV: "development"
      },
      stdio: "inherit"
    }
  );

  devServer.on("exit", (code) => {
    process.exit(code ?? 0);
  });
} else {
  process.env.NODE_ENV = "production";
  process.env.HOSTNAME = host;
  process.env.PORT = port;

  const serverEntry = path.join(webRoot, "apps", "web", "server.js");
  process.chdir(path.dirname(serverEntry));
  require(serverEntry);
}
