# Continue Analysis on Generated Charts

While TableGPT2 excels in data analysis tasks, it currently lacks built-in support for visual modalities. Many data analysis tasks involve visualization, so to address this limitation, we provide an interface for integrating your own Visual Language Model (VLM) plugin.

When the agent performs a visualization task—typically using `matplotlib.pyplot.show`—the VLM will take over from the LLM, offering a more nuanced summarization of the visualization. This approach avoids the common pitfalls of LLMs in visualization tasks, which often either state, "I have plotted the data," or hallucinating the content of the plot.

We continue using the agent from the previous section to perform a data visualization task and observe its final output.

```pycon
>>> human_message = HumanMessage(content="Draw a pie chart based on gender and the number of people of each gender.")

>>> async for event in agent.astream_events(
...     input={
...         "messages": [human_message],
...         "parent_id": "some-parent-id2",
...         "date": date.today(),
...     },
...     version="v2",
...     # We configure the same thread_id to use checkpoints to retrieve the memory of the last run.
...     config={"configurable": {"thread_id": "some-thread-id"}},
... ):
...     evt: str = event["event"]
...     if evt == "on_chat_model_end":
...         print(event["data"]["output"])
...     elif event["name"] == "tools" and evt == "on_chain_stream":
...         for lc_msg in event["data"]["chunk"]["messages"]:
...             print(lc_msg)
...     else:
...         # Other events can be handled here.
...         pass
content="为了绘制性别比例的饼图，我需要计算每个性别的乘客数量。然后，使用 `seaborn` 或 `matplotlib` 绘制饼图。\n\n```python\nimport matplotlib.pyplot as plt\n\n# 计算每个性别的乘客数量\ngender_counts = df['Sex'].value_counts()\n\n# 绘制饼图\nplt.figure(figsize=(8, 8))\nplt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightcoral'])\nplt.title('Gender Distribution')\nplt.show()\n```\n" additional_kwargs={} response_metadata={'finish_reason': 'stop', 'model_name': 'tablegpt2-7b'} id='run-43b661f5-738a-49a1-9b7f-058617290ee5'
content=[{'type': 'text', 'text': '```pycon\n<Figure size 800x800 with 1 Axes>\n```'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgoA...'}}] name='python' id='df24ade2-5514-4fae-a6bd-98e4aae70a85' tool_call_id='e4cdf0fd-dbab-4f80-8dda-e22bb6b2aa03' artifact=[]
content='性别比例的饼图已经成功绘制。' additional_kwargs={} response_metadata={'finish_reason': 'stop', 'model_name': 'tablegpt2-7b'} id='run-a30ff474-79de-4531-94a2-2d3d18b3ae82'
```

We create a new agent and pass it a vlm instance:

```pycon
>>> vlm = ChatOpenAI(openai_api_base="YOUR_VLM_URL", openai_api_key="whatever", model_name="your-vlm-model-name")

>>> agent_with_vlm = create_tablegpt_graph(
...     llm=llm,
...     pybox_manager=pybox_manager,
...     vlm=vlm,
...     checkpointer=memory_saver,
...     session_id="some-session-id",
... )
```

Now we send the same question to the model via agent_with_vlm:

```pycon
>>> async for event in agent_with_vlm.astream_events(
...     input={
...         "messages": [human_message],
...         "parent_id": "some-parent-id4",
...         "date": date.today(),
...     },
...     version="v2",
...     config={"configurable": {"thread_id": "some-thread-id"}},
... ):
...     evt: str = event["event"]
...     if evt == "on_chat_model_end":
...         print(event["data"]["output"])
...     elif event["name"] == "tools" and evt == "on_chain_stream":
...         for lc_msg in event["data"]["chunk"]["messages"]:
...             print(lc_msg)
...     else:
...         # Other events can be handled here.
...         pass

content="为了绘制基于性别的乘客数量的饼图，我们需要先计算每个性别的乘客数量，然后使用 `matplotlib` 来绘制饼图。\n```python\nimport matplotlib.pyplot as plt\n\n# 计算每个性别的乘客数量\ngender_counts = df['Sex'].value_counts()\n\n# 绘制饼图\nplt.figure(figsize=(8, 8))\nplt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)\nplt.title('Gender Distribution of Passengers')\nplt.show()\n```\n" additional_kwargs={} response_metadata={'finish_reason': 'stop', 'model_name': 'tablegpt2-7b'} id='run-bb9405fa-4940-4a64-b5e4-fdc5f54efbcd'
content=[{'type': 'text', 'text': '```pycon\n<Figure size 800x800 with 1 Axes>\n```'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA...'}}] name='python' id='9232614f-2b06-4193-ad08-fed0ab27db4c' tool_call_id='b2cc0ea1-a830-49f1-b848-635ac4311871' artifact=[]
content='饼图已经生成，展示了乘客数据集的性别分布情况。在这个数据集中，男性和女性乘客人数各占 50%，各占半个圆。' additional_kwargs={} response_metadata={'finish_reason': 'stop', 'model_name': 'qwen2-vl-7b-instruct'} id='run-2ef5b082-ea29-4781-a3e8-15c321c93fb1'
```

We observe that the answer provided by the agent with VLM support is significantly more detailed, including a comprehensive description of the generated images.

<details>

<summary>Full code</summary>

```python
import asyncio
from datetime import date
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pybox import LocalPyBoxManager
from tablegpt.agent import create_tablegpt_graph
from tablegpt.agent.file_reading import Stage


class Attachment(TypedDict):
    """Contains at least one dictionary with the key filename."""

    filename: str
    """The dataset uploaded in this session can be a filename, file path, or object storage address."""


# tablegpt-agent fully supports async invocation
async def main() -> None:
    llm = ChatOpenAI(openai_api_base="YOUR_VLLM_URL", openai_api_key="whatever", model_name="TableGPT2-7B")
    vlm = ChatOpenAI(openai_api_base="YOUR_VLM_URL", openai_api_key="whatever", model_name="YOUR_VLM_MODEL_NAME")

    # Use local pybox manager for development and testing
    pybox_manager = LocalPyBoxManager()

    agent_with_vlm = create_tablegpt_graph(
        llm=llm,
        pybox_manager=pybox_manager,
        # We use MemorySaver as a checkpointer to record memory automatically.
        # See <https://langchain-ai.github.io/langgraph/concepts/persistence>
        checkpointer=MemorySaver(),
        vlm=vlm,
        # All code generated in this run will be executed in the kernel with kernel_id 'some-session-id'.
        session_id="some-session-id",
    )

    attachment_msg = HumanMessage(
        content="",
        # The dataset can be viewed in examples/datasets/titanic.csv.
        additional_kwargs={"attachments": [Attachment(filename="titanic.csv")]},
    )
    await agent_with_vlm.ainvoke(
        input={
            "entry_message": attachment_msg,
            "processing_stage": Stage.UPLOADED,
            "messages": [attachment_msg],
            "parent_id": "some-parent-id1",
            "date": date.today(),  # noqa: DTZ011
        },
        config={
            "configurable": {"thread_id": "some-thread-id"},
        },
    )

    human_message = HumanMessage(content="Draw a pie chart based on gender and the number of people of each gender.")

    async for event in agent_with_vlm.astream_events(
        input={
            # After using checkpoint, you only need to add new messages here.
            "messages": [human_message],
            "parent_id": "some-parent-id2",
            "date": date.today(),  # noqa: DTZ011
        },
        version="v2",
        # We configure the same thread_id to use checkpoints to retrieve the memory of the last run.
        config={"configurable": {"thread_id": "some-thread-id"}},
    ):
        print(event)  # noqa: T201


asyncio.run(main())
```

</details>

