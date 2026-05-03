from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

_pydantic_ai = types.ModuleType("pydantic_ai")
_models = types.ModuleType("pydantic_ai.models")
_models_openai = types.ModuleType("pydantic_ai.models.openai")
_providers = types.ModuleType("pydantic_ai.providers")
_providers_openai = types.ModuleType("pydantic_ai.providers.openai")


class _Agent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _OpenAIChatModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _OpenAIProvider:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


_pydantic_ai.Agent = _Agent
_models_openai.OpenAIChatModel = _OpenAIChatModel
_providers_openai.OpenAIProvider = _OpenAIProvider

sys.modules.setdefault("pydantic_ai", _pydantic_ai)
sys.modules.setdefault("pydantic_ai.models", _models)
sys.modules.setdefault("pydantic_ai.models.openai", _models_openai)
sys.modules.setdefault("pydantic_ai.providers", _providers)
sys.modules.setdefault("pydantic_ai.providers.openai", _providers_openai)

from agents import genie_sim  # type: ignore  # noqa: E402


class GenieSimAgentPromptTest(unittest.TestCase):
    def test_agent_instructions_cover_known_scene_codegen_failures(self):
        instructions = genie_sim.GENIE_SIM_AGENT_INSTRUCTIONS

        self.assertIn("Search assets first", instructions)
        self.assertIn("from helper import *", instructions)
        self.assertIn("Never call .add(...) on keywords, Shape, or any Python list", instructions)
        self.assertIn("Call extra registered scene functions via library_call", instructions)
        self.assertIn("Copy this skeleton and replace asset ids, keywords, positions, and support relationships:", instructions)
        self.assertIn("def place_on_top(obj: Shape, support: Shape, xy: tuple[float, float], gap: float = 0.0) -> Shape:", instructions)
        self.assertIn("Registered helper pattern:", instructions)
        self.assertIn('mug = library_call("place_asset"', instructions)


if __name__ == "__main__":
    unittest.main()
