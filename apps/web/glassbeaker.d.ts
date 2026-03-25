export {};

declare global {
  interface GlassBeakerRuntime {
    readonly isElectron: boolean;
    readonly electron: string;
    readonly chrome: string;
    readonly node: string;
    readonly packaged: boolean;
  }

  interface Window {
    glassBeaker?: GlassBeakerRuntime;
  }
}
