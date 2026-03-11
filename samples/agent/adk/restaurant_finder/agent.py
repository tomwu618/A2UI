# Copyright 2025 Google LLC
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
from collections.abc import AsyncIterable
from typing import Any

import jsonschema
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    DataPart,
    Part,
    TextPart,
)
from prompt_builder import (
    get_text_prompt,
    ROLE_DESCRIPTION,
    UI_DESCRIPTION,
)
from tools import get_restaurants
from a2ui.core.schema.constants import VERSION_0_8, A2UI_OPEN_TAG, A2UI_CLOSE_TAG
from a2ui.core.schema.manager import A2uiSchemaManager
from a2ui.core.parser.parser import parse_response, ResponsePart
from a2ui.basic_catalog.provider import BasicCatalog
from a2ui.core.schema.common_modifiers import remove_strict_validation
from a2ui.a2a import create_a2ui_part, get_a2ui_agent_extension, parse_response_to_parts

logger = logging.getLogger(__name__)


class RestaurantAgent:
  """An agent that finds restaurants based on user criteria."""

  SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

  def __init__(self, base_url: str, use_ui: bool = False):
    self.base_url = base_url
    self.use_ui = use_ui
    self._schema_manager = (
        A2uiSchemaManager(
            VERSION_0_8,
            catalogs=[
                BasicCatalog.get_config(version=VERSION_0_8, examples_path="examples")
            ],
            schema_modifiers=[remove_strict_validation],
        )
        if use_ui
        else None
    )
    
    self.api_key = os.environ.get("OPENAI_API_KEY")
    self.base_api_url = os.environ.get("OPENAI_BASE_URL")
    self.model_name = os.environ.get("LLM_MODEL_NAME", "gpt-4o")
    
    self.client = AsyncOpenAI(
        api_key=self.api_key,
        base_url=self.base_api_url,
    )
    
    self.instruction = (
        self._schema_manager.generate_system_prompt(
            role_description=ROLE_DESCRIPTION,
            ui_description=UI_DESCRIPTION,
            include_schema=True,
            include_examples=True,
            validate_examples=True,
        )
        if use_ui
        else get_text_prompt()
    )
    
    self._sessions = {} # Simple in-memory session management for now

  def get_agent_card(self) -> AgentCard:
    capabilities = AgentCapabilities(
        streaming=True,
        extensions=[
            get_a2ui_agent_extension(
                self._schema_manager.accepts_inline_catalogs,
                self._schema_manager.supported_catalog_ids,
            )
        ],
    )
    skill = AgentSkill(
        id="find_restaurants",
        name="Find Restaurants Tool",
        description=(
            "Helps find restaurants based on user criteria (e.g., cuisine, location)."
        ),
        tags=["restaurant", "finder"],
        examples=["Find me the top 10 chinese restaurants in the US"],
    )

    return AgentCard(
        name="Restaurant Agent",
        description="This agent helps find restaurants based on user criteria.",
        url=self.base_url,
        version="1.0.0",
        default_input_modes=RestaurantAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=RestaurantAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )

  def get_processing_message(self) -> str:
    return "Finding restaurants that match your criteria..."

  def _get_tools(self) -> list[dict[str, Any]]:
      return [
          {
              "type": "function",
              "function": {
                  "name": "get_restaurants",
                  "description": "Call this tool to get a list of restaurants based on user criteria (cuisine, location, etc.).",
                  "parameters": {
                      "type": "object",
                      "properties": {
                          "cuisine": {
                              "type": "string",
                              "description": "The type of cuisine to search for (e.g., Italian, Chinese)."
                          },
                          "location": {
                              "type": "string",
                              "description": "The location to search in (e.g., San Francisco, New York)."
                          },
                          "query": {
                              "type": "string",
                              "description": "A general search query for the restaurant."
                          }
                      },
                      "required": ["query"]
                  }
              }
          }
      ]

  async def stream(self, query, session_id) -> AsyncIterable[dict[str, Any]]:
    if session_id not in self._sessions:
        self._sessions[session_id] = []
    
    messages = self._sessions[session_id]
    messages.append({"role": "user", "content": query})

    # --- Begin: UI Validation and Retry Logic ---
    max_retries = 1  # Total 2 attempts
    attempt = 0
    current_query_text = query

    # Ensure schema was loaded
    selected_catalog = self._schema_manager.get_selected_catalog() if self._schema_manager else None
    if self.use_ui and (not selected_catalog or not selected_catalog.catalog_schema):
      logger.error(
          "--- RestaurantAgent.stream: A2UI_SCHEMA is not loaded. "
          "Cannot perform UI validation. ---"
      )
      yield {
          "is_task_complete": True,
          "parts": [
              Part(
                  root=TextPart(
                      text=(
                          "I'm sorry, I'm facing an internal configuration error with"
                          " my UI components. Please contact support."
                      )
                  )
              )
          ],
      }
      return

    while attempt <= max_retries:
      attempt += 1
      logger.info(
          f"--- RestaurantAgent.stream: Attempt {attempt}/{max_retries + 1} "
          f"for session {session_id} ---"
      )

      final_response_content = None
      
      yield {
          "is_task_complete": False,
          "updates": self.get_processing_message(),
      }

      try:
          response = await self.client.chat.completions.create(
              model=self.model_name,
              messages=[{"role": "system", "content": self.instruction}] + messages,
              tools=self._get_tools(),
              tool_choice="auto",
          )
          
          message = response.choices[0].message
          
          if message.tool_calls:
              for tool_call in message.tool_calls:
                  if tool_call.function.name == "get_restaurants":
                      args = json.loads(tool_call.function.arguments)
                      # Mock ToolContext for the tool
                      class MockToolContext:
                          def __init__(self, base_url):
                              self.state = {"base_url": base_url}
                      
                      tool_result = get_restaurants(
                          query=args.get("query"),
                          cuisine=args.get("cuisine"),
                          location=args.get("location"),
                          tool_context=MockToolContext(self.base_url)
                      )
                      
                      messages.append(message)
                      messages.append({
                          "role": "tool",
                          "tool_call_id": tool_call.id,
                          "name": "get_restaurants",
                          "content": tool_result
                      })
                      
                      # Call again with tool results
                      response = await self.client.chat.completions.create(
                          model=self.model_name,
                          messages=[{"role": "system", "content": self.instruction}] + messages,
                      )
                      final_response_content = response.choices[0].message.content
          else:
              final_response_content = message.content

      except Exception as e:
          logger.error(f"Error calling OpenAI: {e}")
          final_response_content = f"Error: {e}"

      if final_response_content is None:
        logger.warning(
            "--- RestaurantAgent.stream: Received no final response content from"
            f" runner (Attempt {attempt}). ---"
        )
        if attempt <= max_retries:
          messages.append({"role": "user", "content": "I received no response. Please try again."})
          continue  # Go to next retry
        else:
          final_response_content = (
              "I'm sorry, I encountered an error and couldn't process your request."
          )

      is_valid = False
      error_message = ""

      if self.use_ui:
        logger.info(
            "--- RestaurantAgent.stream: Validating UI response (Attempt"
            f" {attempt})... ---"
        )
        try:
          response_parts = parse_response(final_response_content)

          for part in response_parts:
            if not part.a2ui_json:
              continue

            parsed_json_data = part.a2ui_json
            selected_catalog.validator.validate(parsed_json_data)
            is_valid = True

        except (
            ValueError,
            json.JSONDecodeError,
            jsonschema.exceptions.ValidationError,
        ) as e:
          logger.warning(
              f"--- RestaurantAgent.stream: A2UI validation failed: {e} (Attempt"
              f" {attempt}) ---"
          )
          error_message = f"Validation failed: {e}."

      else:  # Not using UI, so text is always "valid"
        is_valid = True

      if is_valid:
        messages.append({"role": "assistant", "content": final_response_content})
        final_parts = parse_response_to_parts(
            final_response_content, fallback_text="OK."
        )

        yield {
            "is_task_complete": True,
            "parts": final_parts,
        }
        return

      if attempt <= max_retries:
        retry_msg = (
            f"Your previous response was invalid. {error_message} You MUST generate a"
            " valid response that strictly follows the A2UI JSON SCHEMA. The response"
            " MUST be a JSON list of A2UI messages. Ensure each JSON part is wrapped in"
            f" '{A2UI_OPEN_TAG}' and '{A2UI_CLOSE_TAG}' tags. Please retry the"
            f" original request."
        )
        messages.append({"role": "user", "content": retry_msg})

    yield {
        "is_task_complete": True,
        "parts": [
            Part(
                root=TextPart(
                    text=(
                        "I'm sorry, I'm having trouble generating the interface for"
                        " that request right now. Please try again in a moment."
                    )
                )
            )
        ],
    }
    # --- End: UI Validation and Retry Logic ---
