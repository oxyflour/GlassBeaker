import { cp, mkdir, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webRoot = path.resolve(__dirname, "..");
const nextDir = path.join(webRoot, ".next");
const standaloneDir = path.join(nextDir, "standalone");
const standaloneStaticDir = path.join(standaloneDir, ".next", "static");
const publicDir = path.join(webRoot, "public");
const staticDir = path.join(nextDir, "static");

async function copyDir(from: string, to: string): Promise<void> {
  await rm(to, { recursive: true, force: true });
  await mkdir(path.dirname(to), { recursive: true });
  await cp(from, to, { recursive: true });
}

async function main(): Promise<void> {
  await copyDir(publicDir, path.join(standaloneDir, "public"));
  await copyDir(staticDir, standaloneStaticDir);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
