# GlassBeaker

Minimal `Next.js + Electron` workspace managed by `pnpm`.

## Commands

- `pnpm install`
- `pnpm dev`
- `pnpm build`
- `pnpm run package:dir`
- `pnpm dist`

## Structure

- `apps/web`: Next.js app with `output: 'standalone'`
- `apps/desktop`: Electron shell that starts the Next server in a `utilityProcess`

