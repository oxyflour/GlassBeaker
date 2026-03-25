const path = require("node:path");
const { spawn } = require("node:child_process");

const electronBinary = require("electron");
const desktopRoot = path.resolve(__dirname, "..");
const env = { ...process.env };

delete env.ELECTRON_RUN_AS_NODE;

const child = spawn(electronBinary, ["."], {
  cwd: desktopRoot,
  env,
  stdio: "inherit"
});

child.on("exit", (code) => {
  process.exit(code ?? 0);
});
