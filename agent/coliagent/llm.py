"""Model backends.

``OpenAICompatible`` talks to any /v1/chat/completions server; the default URL
is a local Colibri gateway (``coli serve``), so the agent can run entirely on
the machine it is improving. ``ScriptedBackend`` replays canned replies for
tests and dry runs.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional


class LLMError(Exception):
    pass


class OpenAICompatible:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: float = 600.0,
                 temperature: float = 0.2, max_tokens: int = 2048):
        self.base_url = (base_url or os.environ.get("COLIAGENT_BASE_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = model or os.environ.get("COLIAGENT_MODEL") or "colibri"
        self.api_key = api_key if api_key is not None else os.environ.get("COLIAGENT_API_KEY", "")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: List[Dict[str, str]], **kw) -> str:
        body = {"model": self.model, "messages": messages, "temperature": kw.get("temperature", self.temperature),
                "max_tokens": kw.get("max_tokens", self.max_tokens), "stream": False}
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise LLMError(f"{self.base_url}: {e}") from e
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected reply shape: {str(data)[:200]}") from e


class ScriptedBackend:
    def __init__(self, replies: Iterable[str]):
        self.replies = list(replies)
        self.calls: List[List[Dict[str, str]]] = []

    def complete(self, messages: List[Dict[str, str]], **kw) -> str:
        self.calls.append(messages)
        if not self.replies:
            raise LLMError("scripted backend has no replies left")
        return self.replies.pop(0)
