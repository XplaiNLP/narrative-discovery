import asyncio
import json

import backoff
import httpx
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)
from pydantic import BaseModel


class ClaimList(BaseModel):
    claims: list[str]


class SemanticGroup(BaseModel):
    semantic_unit: str
    entities: list[str]
    relationships: list[str]


class TextDecomposition(BaseModel):
    Output: list[SemanticGroup]


class RelationshipReconstruction(BaseModel):
    source: str
    relationship: str
    target: str


class GroupSummary(BaseModel):
    title: str
    description: str


class GroupSummaryResponse(BaseModel):
    group_summaries: list[GroupSummary]


class NarrativeResponse(BaseModel):
    response: str


def _clean_json(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
    last = content.rfind("}")
    if last != -1:
        content = content[: last + 1]
    return content.strip()


class LLMClient:
    def __init__(self, provider, model, api_key, api_base=None, concurrency=16,
                 max_tokens=10_000, temperature=0.0, timeout=600.0):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(concurrency)

        kwargs = {"api_key": api_key or "EMPTY", "max_retries": 0, "timeout": timeout}
        if api_base:
            kwargs["base_url"] = api_base
        kwargs["http_client"] = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=concurrency + 10,
                max_keepalive_connections=concurrency,
            ),
            timeout=httpx.Timeout(timeout, connect=30.0),
        )
        self.client = AsyncOpenAI(**kwargs)

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APITimeoutError, APIConnectionError, json.JSONDecodeError),
        max_time=300, max_tries=5,
    )
    async def _call(self, query, response_format=None, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if response_format is not None:
            if self.provider == "openai":
                params["response_format"] = response_format
                resp = await self.client.beta.chat.completions.parse(**params)
                return json.loads(resp.choices[0].message.parsed.model_dump_json())
            params["extra_body"] = {
                "structured_outputs": {"json": response_format.model_json_schema()},
                "chat_template_kwargs": {"enable_thinking": False},
            }
            resp = await self.client.chat.completions.create(**params)
            return json.loads(_clean_json(resp.choices[0].message.content))
        if self.provider == "vllm":
            params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        resp = await self.client.chat.completions.create(**params)
        return (resp.choices[0].message.content or "").strip()

    async def call(self, query, response_format=None, system_prompt=None):
        async with self.semaphore:
            try:
                return await self._call(query, response_format, system_prompt)
            except Exception as e:
                print(f"  LLM call failed: {type(e).__name__}: {e}")
                return None
