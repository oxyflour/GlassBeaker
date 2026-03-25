const { contextBridge } = require("electron");

const packagedArg = process.argv.find((arg) =>
  arg.startsWith("--glassbeaker-packaged=")
);

contextBridge.exposeInMainWorld("glassBeaker", {
  isElectron: true,
  electron: process.versions.electron,
  chrome: process.versions.chrome,
  node: process.versions.node,
  packaged: packagedArg === "--glassbeaker-packaged=1"
});
