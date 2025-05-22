from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .rate_limiter import RateLimiter

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass(slots=True)
class LLMTask:
    """Represents a single LLM call to be executed."""

    model_name: str
    estimate_in: int
    estimate_out: int
    fn: Callable[[], Awaitable[Any]]


class LLMTaskRunner:
    """Simple scheduler that runs tasks with per-model queues."""

    def __init__(self, rate_limiter: RateLimiter | None = None, *, max_workers_per_model: int = 4) -> None:
        self._queues: dict[str, asyncio.Queue[LLMTask]] = defaultdict(asyncio.Queue)
        self._sems: dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(max_workers_per_model))
        self._rate_limiter = rate_limiter or RateLimiter()
        self._max_workers = max_workers_per_model

    def add_tasks(self, tasks: list[LLMTask]) -> None:
        """Add a batch of tasks to their model-specific queues."""
        for task in tasks:
            self._queues[task.model_name].put_nowait(task)

    async def _model_worker(self, model_name: str) -> None:
        queue = self._queues[model_name]
        sem = self._sems[model_name]
        while True:
            task = await queue.get()
            if not self._rate_limiter.has_capacity(
                model_name=task.model_name,
                desired_input_tokens=task.estimate_in,
                desired_output_tokens=task.estimate_out,
            ):
                queue.put_nowait(task)
                queue.task_done()
                await asyncio.sleep(0.1)
                if queue.empty() and sem._value == self._max_workers:
                    break
                continue
            async with sem:
                try:
                    await task.fn()
                finally:
                    queue.task_done()
            if queue.empty() and sem._value == self._max_workers:
                break

    async def run(self) -> None:
        """Run tasks for all known models and wait for completion."""
        async with asyncio.TaskGroup() as tg:
            for model in list(self._queues):
                tg.create_task(self._model_worker(model))
