"""
Pydantic AI agent for GP Connect expert.
"""

from __future__ import annotations as _annotations
from typing import List

import os

from dataclasses import dataclass
from dotenv import load_dotenv
import httpx

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel

# from pydantic_ai.models.gemini import GeminiModel
from openai import AsyncOpenAI
from supabase import Client

load_dotenv()

llm = os.getenv("LLM_MODEL", "gpt-4o-mini")
model = OpenAIModel(llm)
# model = GeminiModel('gemini-2.0-flash')


@dataclass
class PydanticAIDeps:
    """Dependencies for the Pydantic AI agent."""

    supabase: Client
    openai_client: AsyncOpenAI


SYSTEM_PROMPT = """
You are an expert on GP Connect - a standards based health service allowing
health care professionals to access patient's GP records. You have access to
the documentation including examples, an API reference, and other resources to
help you answer questions about the service and standards.

Your only job is to assist with this and you don't answer other questions
besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you
look at the documentation with the provided tools before answering the user's
question unless you have already.

If the user's query contains words or terms that are lowercased but would
normally be capitalised e.g. 'SNOMED' and 'snomed', replace the term with the
most common capitalisation.

When providing an answer to the user, always return the source URL of the
page(s) where the answer came from.

Always let the user know when you didn't find the answer in the documentation
- be honest.
"""

gp_connect_agent = Agent(
    model,
    system_prompt=SYSTEM_PROMPT,
    deps_type=PydanticAIDeps,
    retries=2,
)


async def get_embedding(user_query: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        print(f"Getting embedding for: {user_query}")
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=user_query
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


@gp_connect_agent.tool
async def retrieve_relevant_documentation(
    ctx: RunContext[PydanticAIDeps], user_query: str
) -> str:
    """
    Retrieve relevant documentation pages and their source URL, based on the
    user's query. Replace 'snomed' with 'snomed ct' in the query.

    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's query

    Returns:
        A formatted string containing the most relevant documentation pages and
        their source URL.
    """
    try:
        # Get the embedding for the query. The query is deteremined by the LLM
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            "match_site_pages",
            {
                "query_embedding": query_embedding,
                "match_count": 3,
                "filter": {"source": "gpconnect-1-6-0"},
            },
        ).execute()

        print(f"Results returned: {len(result.data)}.")

        if not result.data:
            return "No relevant documentation found."

        # Combine the pages
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}

Source URL: {doc['url']}
"""
            formatted_chunks.append(chunk_text)

        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"
