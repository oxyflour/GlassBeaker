from copilotkit import CopilotKitMiddleware, LangGraphAGUIAgent
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

from agents.chat.model import model

agent = create_deep_agent(
    model=model,
    middleware=[CopilotKitMiddleware()], #type: ignore
    system_prompt="You are a helpful research assistant.",
    checkpointer=MemorySaver()
)

agent = LangGraphAGUIAgent(
    name="sample_agent",
    description="An example agent to use as a starting point for your own agent.",
    graph=agent,
)
