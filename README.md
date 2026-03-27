# GlassBeaker

Minimal `Next.js + Electron + Python` workspace managed by `pnpm`.

## Commands

- `pnpm install`
- `pnpm dev`
- `pnpm build`
- `pnpm run package:dir`
- `pnpm dist`

`pnpm dev` now starts both the standalone Next server and the Python backend through the Electron main process. `pnpm run package:dir` and `pnpm dist` build a single-file Python executable before invoking `electron-builder`.
If you need a non-default Python origin, set `GLASSBEAKER_PYTHON_ORIGIN` or `GLASSBEAKER_PYTHON_HOST`/`GLASSBEAKER_PYTHON_PORT` before `pnpm build`.

## Structure

- `apps/web`: Next.js app with `output: 'standalone'` and a fallback `/api` rewrite to Python
- `apps/desktop`: Electron shell that starts the Next server and the Python service in a `utilityProcess`
- `apps/python`: FastAPI/Uvicorn service, packaged into a single-file executable for desktop builds
