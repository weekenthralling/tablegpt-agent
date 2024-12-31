import asyncio
import datetime
import json
import logging
import traceback
from typing import Any, AsyncContextManager, TYPE_CHECKING

import aiofiles
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from tqdm.asyncio import tqdm

from .evaluator import create_evaluator_runnable
from .evaluatee import create_evaluatee_runnable
from .workflow import create_eval_workflow

if TYPE_CHECKING:
    from langgraph.checkpoint.base import Checkpoint

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        queue: asyncio.Queue,
        stop_event: asyncio.Event | None = None,
        pbar: tqdm | None = None,
        evaluator_config: dict[str, Any] = {},
        lifespan: AsyncContextManager | None = None,
    ) -> None:
        self.queue = queue
        self.stop_event = stop_event
        self.pbar = pbar
        self.evaluator_config = evaluator_config
        self.lifespan = lifespan

    async def run(self) -> None:
        logger.info("Worker started")
        async with self.lifespan() as context:
            while self.stop_event is None or not self.stop_event.is_set():
                try:
                    payload = self.queue.get_nowait()
                    executor = TableGPTEvalExecutor(evaluator_config=self.evaluator_config, evaluatee_context=context)
                    await executor.run(payload=payload)
                    if self.pbar is not None:
                        self.pbar.update(1)
                except asyncio.QueueEmpty:
                    # No more tasks in the queue, quit current worker
                    logger.info("Worker finished")
                    break
                except Exception:
                    logger.exception("Worker encountered an error")
                    # Set the stop event to cancel other workers
                    if self.stop_event is not None:
                        self.stop_event.set()
                    break


class AbstractEvalExecutor:
    ...


class TableGPTEvalExecutor(AbstractEvalExecutor):
    def __init__(self, evaluator_config: dict, evaluatee_context: dict[str, Any] | None = None) -> None:
        self.evaluator = create_evaluator_runnable(ChatOpenAI(**evaluator_config))
        self.evaluatee_context = evaluatee_context if evaluatee_context is not None else {}
        self.eval_run_output_file = f"eval_run_{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d_%H%M%S')}.jsonl"

    async def run(self, payload: dict[str, Any]) -> None:
        """Run the evaluation workflow.
        Usually a evaluatee runnable will be executed, followed by a evaluator runnable.

        Args:
            payload (dict[str, Any]): Evaluation payload.
        """
        logger.debug("Evaluating sample: %s", payload)
        item: dict[str, Any] = payload["item"]

        checkpointer = MemorySaver()
        evaluatee = await create_evaluatee_runnable(
            datasets=payload.get("datasets"),
            checkpointer=checkpointer,
            **self.evaluatee_context,
        )
        self.eval_wf = create_eval_workflow(evaluatee=evaluatee, evaluator=self.evaluator)
        criteria = payload.get("criteria")
        try:
            res = await self.eval_wf.ainvoke(
                input={
                    "input": item["input"],
                    "reference_answer": item["expected_output"],
                    "criteria": criteria,
                    "redlines": payload.get("redlines", []),
                },
            )
            evaluation = res["evaluation"]
            evaluatee_answer = res["evaluatee_answer"]
        except Exception:
            logger.exception(
                "Evaluation Workflow failed, item: %s, context: %s",
                item["input"],
                self.evaluatee_context,
            )
            # We treat any exception in agent invocation as a bad case
            err_info = traceback.format_exc()
            evaluation = {
                "score": 0,
                "explaination": err_info,
            }
            evaluatee_answer = ""

        # TODO: get rid of the 'session_id'. It's a evaluatee thing.
        checkpoint: Checkpoint = checkpointer.get(
            config={
                "configurable": {"thread_id": self.evaluatee_context["session_id"]},
            }
        )
        messages = checkpoint["channel_values"].get("messages", [])
        messages = [message.dict() for message in messages]

        eval_result = {
            "input": item["input"],
            "evaluation": evaluation,
            "reference_answer": item["expected_output"],
            "evaluatee_answer": evaluatee_answer,
            "criteria": criteria,
            "redlines": payload.get("redlines", []),
            "messages": messages,
        }

        async with aiofiles.open(self.eval_run_output_file, mode="a") as f:
            await f.write(json.dumps(eval_result, ensure_ascii=False) + "\n")
