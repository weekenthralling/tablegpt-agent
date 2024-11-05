import asyncio
from datetime import date

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tablegpt.agent import create_tablegpt_graph
from pybox import LocalPyBoxManager


# tablegpt-agent fully supports async invocation
async def main() -> None:
  llm = ChatOpenAI(openai_api_base="YOUR_VLLM_URL", openai_api_key="whatever", model_name="TableGPT2-7B")

  # Use local pybox manager for development and testing
  pybox_manager = LocalPyBoxManager()

  agent = create_tablegpt_graph(
    llm=llm,
    pybox_manager=pybox_manager,
  )

  message = HumanMessage(content="Hi")
  input = {
      "messages": [message],
      "parent_id": "some-parent-id",
      "date": date.today(),
  }


  async for event in agent.astream_events(
      input=input,
      version="v2",
  ):
      print(event)


asyncio.run(main())
