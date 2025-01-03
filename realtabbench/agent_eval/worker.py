from __future__ import annotations

import asyncio
import json
import logging
import traceback
from typing import TYPE_CHECKING, Any

import aiofiles
from langchain_openai import ChatOpenAI
from tqdm.asyncio import tqdm

from .evaluator import create_evaluator_runnable
from .evaluator.prompt import (DEFAULT_CRITERIA_WITH_REFERENCE_ANSWER,
                               DEFAULT_CRITERIA_WITHOUT_REFERENCE_ANSWER)

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from .evaluatee import AbstractEvaluatee

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        queue: asyncio.Queue,
        evaluatee: AbstractEvaluatee,
        stop_event: asyncio.Event | None = None,
        pbar: tqdm | None = None,
        evaluator_config: dict[str, Any] = {},
        eval_run_output_file: str = "eval-result.jsonl",
    ) -> None:
        self.queue = queue
        self.evaluatee = evaluatee
        self.stop_event = stop_event
        self.pbar = pbar
        self.evaluator_config = evaluator_config
        self.eval_run_output_file = eval_run_output_file

    async def run(self) -> None:
        logger.info("Worker started")
        async with self.evaluatee:
            while self.stop_event is None or not self.stop_event.is_set():
                try:
                    sample = self.queue.get_nowait()
                    executor = EvalExecutor(self.evaluatee, self.evaluator_config, self.eval_run_output_file)
                    await executor.run(sample)
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


class EvalExecutor:
    def __init__(
        self,
        evaluatee: AbstractEvaluatee,
        evaluator_config: dict[str, Any],
        eval_run_output_file: str = "eval-result.jsonl",
    ) -> None:
        self.evaluator = create_evaluator_runnable(ChatOpenAI(**evaluator_config))
        self.evaluatee = evaluatee
        self.eval_run_output_file = eval_run_output_file

    async def run(self, sample: BaseMessage) -> None:
        """Run the evaluation workflow.
        Usually a evaluatee runnable will be executed, followed by a evaluator runnable.

        Args:
            sample (BaseMessage): Evaluation sample.
        """
        logger.debug("Evaluating sample: %s", sample)
        criteria = (
            sample.additional_kwargs.get("criteria")
            if sample.additional_kwargs.get("criteria")
            else (
                DEFAULT_CRITERIA_WITH_REFERENCE_ANSWER
                if sample.additional_kwargs.get("expected_output")
                else DEFAULT_CRITERIA_WITHOUT_REFERENCE_ANSWER
            )
        )
        reference_answer = sample.additional_kwargs.get("expected_output")
        redlines = sample.additional_kwargs.get("redlines", [])
        try:
            messages = await self.evaluatee(sample)
            evaluatee_answer = messages[-1].content
            evaluation = await self.evaluator.ainvoke(
                input={
                    "question": sample.content,
                    "reference_answer": reference_answer,
                    "answer": evaluatee_answer,
                    "criteria": criteria,
                    "redlines": redlines,
                },
            )
        except Exception:
            logger.exception(
                "Evaluation Workflow failed, item: %s, context: %s",
                sample,
                self.evaluatee.context,
            )
            # We treat any exception in agent invocation as a bad case
            err_info = traceback.format_exc()
            evaluation = {
                "score": 0,
                "explaination": err_info,
            }
            evaluatee_answer = ""
            messages = []

        messages = [message.model_dump() for message in messages]

        eval_result = {
            "input": sample.content,
            "evaluation": evaluation,
            "reference_answer": reference_answer,
            "evaluatee_answer": evaluatee_answer,
            "criteria": criteria,
            "redlines": redlines,
            "messages": messages,
        }

        async with aiofiles.open(self.eval_run_output_file, mode="a") as f:
            await f.write(json.dumps(eval_result, ensure_ascii=False) + "\n")
