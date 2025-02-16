"""
UI for the GP Connect Expert chatbot.
"""

from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import json
import os

import streamlit as st
import logfire
from supabase import Client
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter,
)
from gp_connect_agent import gp_connect_agent, PydanticAIDeps

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

openai_client = AsyncOpenAI()
supabase: Client = Client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire="never")


class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal["user", "model"]
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == "system-prompt":
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == "user-prompt":
        with st.chat_message("user", avatar="😃"):
            st.markdown(part.content)
    # text
    elif part.part_kind == "text":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(part.content)


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = PydanticAIDeps(supabase=supabase, openai_client=openai_client)

    # Clear placeholder and indicate that we're searching
    message_placeholder = st.empty()
    message_placeholder.status("Searching for answer...")
    # Run the agent in a stream
    async with gp_connect_agent.run_stream(
        user_input,
        deps=deps,
        # pass entire conversation so far
        message_history=st.session_state.messages[:-1],
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [
            msg
            for msg in result.new_messages()
            if not (
                hasattr(msg, "parts")
                and any(part.part_kind == "user-prompt" for part in msg.parts)
            )
        ]
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )
        print(f"messages: {st.session_state.messages}")


async def main():
    """Main function to run the Streamlit UI."""
    st.set_page_config(page_title="🔎 GP Connect Chat", page_icon="🏥")
    st.title("🔎 GP Connect Chat")
    st.write("Ask a question about GP Connect Access Record Structured")
    with st.sidebar:
        password = st.text_input("Password", type="password", key="password")
        """
        Quick links:
        - [GP Connect](https://digital.nhs.uk/services/gp-connect)
        - [GP Connect 1.6.0](https://developer.nhs.uk/apis/gpconnect-1-6-0/)
        """

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    if user_input := st.chat_input("Ask a question about GP Connect"):
        if not password:
            st.error("A password is required to continue.")
            st.stop()
        if password != os.getenv("PASSWORD"):
            st.error("Please enter the correct password to continue.")
            st.stop()
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        # Display user prompt in the UI
        with st.chat_message("user", avatar="😃"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant", avatar="🤖"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())
