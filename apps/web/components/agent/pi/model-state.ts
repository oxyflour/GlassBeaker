import {
  type CustomProvider,
  type Model,
  ModelSelector,
  type ThinkingLevel,
} from "@mariozechner/pi-web-ui";
import { useEffect, useState } from "react";

type ModelStateOptions = {
  provider?: CustomProvider;
  thinkingLevel?: ThinkingLevel;
};

export function usePiModelState(options: ModelStateOptions) {
  const [currentModel, setCurrentModel] = useState<Model<any> | undefined>(options.provider?.models?.[0]);
  const [currentThinkingLevel, setCurrentThinkingLevel] = useState<ThinkingLevel>(options.thinkingLevel ?? "off");

  useEffect(() => {
    if (!options.provider) {
      return;
    }

    setCurrentModel((previousModel) =>
      previousModel?.provider === options.provider?.id ? previousModel : options.provider?.models?.[0],
    );
  }, [options.provider]);

  useEffect(() => {
    if (options.thinkingLevel) {
      setCurrentThinkingLevel(options.thinkingLevel);
    }
  }, [options.thinkingLevel]);

  function openModelSelector() {
    if (!currentModel) {
      return;
    }

    ModelSelector.open(currentModel, setCurrentModel, options.provider ? [options.provider.id] : undefined);
  }

  return {
    currentModel,
    currentThinkingLevel,
    openModelSelector,
    setCurrentThinkingLevel,
  };
}
