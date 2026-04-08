import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    os.environ.get('COPILOTKIT_MODEL', "gpt-4o"),
    provider=OpenAIProvider(
        api_key=os.environ.get('OPENAI_API_KEY', ''),
        base_url=os.environ.get('OPENAI_BASE_URL', '')
    ))
agent = Agent(model, instructions='Be fun!')
