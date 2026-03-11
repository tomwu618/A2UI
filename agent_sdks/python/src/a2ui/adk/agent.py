# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from typing import Any, AsyncIterable, Callable, Optional, Union

from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class OpenAiLlmAgent:
    """An LLM agent that uses OpenAI standard API."""

    def __init__(
        self,
        name: str,
        instruction: str,
        tools: Optional[list[Callable]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.name = name
        self.instruction = instruction
        self.tools = tools or []
        self.model = model or os.environ.get("LLM_MODEL_NAME", "gpt-4o")
        
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )
        
        self.openai_tools = self._convert_tools(self.tools)

    def _convert_tools(self, tools: list[Callable]) -> list[dict[str, Any]]:
        openai_tools = []
        for tool in tools:
            if hasattr(tool, "_get_openai_declaration"):
                openai_tools.append(tool._get_openai_declaration())
            elif hasattr(tool, "__name__"):
                # Basic conversion for simple functions if needed
                # For now, we assume tools are either A2UI tools or simple functions
                # that we might need to wrap.
                pass
        return openai_tools

    async def chat(self, messages: list[dict[str, str]]) -> Any:
        full_messages = [{"role": "system", "content": self.instruction}] + messages
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            tools=self.openai_tools if self.openai_tools else None,
            tool_choice="auto" if self.openai_tools else None,
        )
        return response

def convert_a2ui_tool_to_openai(a2ui_tool_schema: dict[str, Any]) -> dict[str, Any]:
    """Converts A2UI tool schema to OpenAI tool format."""
    return {
        "type": "function",
        "function": {
            "name": "render_a2ui_update",
            "description": "Render a rich user interface using A2UI components.",
            "parameters": a2ui_tool_schema
        }
    }
