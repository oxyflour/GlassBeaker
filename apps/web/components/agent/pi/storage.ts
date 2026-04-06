import {
  AppStorage,
  type CustomProvider,
  CustomProvidersStore,
  IndexedDBStorageBackend,
  ProviderKeysStore,
  SessionsStore,
  SettingsStore,
  setAppStorage,
} from "@mariozechner/pi-web-ui";
import { useEffect, useRef } from "react";

type StorageOptions = {
  provider?: CustomProvider;
  settings?: Record<string, any>;
};

function setupStorage(options: StorageOptions) {
  const settings = new SettingsStore();
  const providerKeys = new ProviderKeysStore();
  const sessions = new SessionsStore();
  const customProvider = new CustomProvidersStore();
  const backend = new IndexedDBStorageBackend({
    dbName: "glass-beaker-pi",
    version: 1,
    stores: [
      settings.getConfig(),
      providerKeys.getConfig(),
      sessions.getConfig(),
      customProvider.getConfig(),
      SessionsStore.getMetadataConfig(),
    ],
  });

  settings.setBackend(backend);
  providerKeys.setBackend(backend);
  sessions.setBackend(backend);
  customProvider.setBackend(backend);

  for (const [key, value] of Object.entries(options.settings || {})) {
    settings.set(key, value);
  }

  if (options.provider) {
    customProvider.set(options.provider);
  }

  const storage = new AppStorage(settings, providerKeys, sessions, customProvider, backend);
  setAppStorage(storage);
  return storage;
}

export function usePiStorage(options: StorageOptions) {
  const storageRef = useRef<AppStorage | null>(null);

  useEffect(() => {
    storageRef.current = setupStorage(options);
  }, [options.provider, options.settings]);

  async function ensureStorage() {
    if (!storageRef.current) {
      storageRef.current = setupStorage(options);
    }
    return storageRef.current;
  }

  return { ensureStorage };
}
