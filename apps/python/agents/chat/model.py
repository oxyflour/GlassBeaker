from langchain_openai import ChatOpenAI
import os

model = ChatOpenAI(
    model=os.environ.get('COPILOTKIT_MODEL', "gpt-4o"),
    api_key=os.environ.get('OPENAI_API_KEY'), #type: ignore
    base_url=os.environ.get('OPENAI_BASE_URL')
)