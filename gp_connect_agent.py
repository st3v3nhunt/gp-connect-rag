"""
Pydantic AI agent for GP Connect expert.
"""

from __future__ import annotations as _annotations
from typing import List

# import asyncio
import os

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import httpx

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client

load_dotenv()

llm = os.getenv("LLM_MODEL", "gpt-4o-mini")
model = OpenAIModel(llm)

logfire.configure()
logfire.instrument_httpx(capture_all=True)


@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI


SYSTEM_PROMPT = """
You are an expert on GP Connect - a standards based health service allowing health care professionals to access
patient's GP records. You have access to the documentation including examples, an API reference, and other resources to
help you answer questions about the service and standards.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Include additional information or context to the user's query when calling
tools, if you think it would help return relevant documentation.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided
tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""
# Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

gp_connect_agent = Agent(
    model,
    system_prompt=SYSTEM_PROMPT,
    deps_type=PydanticAIDeps,
    retries=2,
)


async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        print(f"Getting embedding for: {text}")
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        # print(f"Embedding response: {response}")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


@gp_connect_agent.tool
async def retrieve_relevant_documentation(
    ctx: RunContext[PydanticAIDeps], user_query: str
) -> str:
    """
    Retrieve relevant documentation pages based on the user's, unaltered query.

    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query

    Returns:
        A formatted string containing the top 3 most relevant documentation pages
    """
    try:
        # Get the embedding for the query
        # TODO: The value of the user query is being altered before it goes to
        # get the embedding which is causing inconsistent results
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

        print(f"Result length: {len(result.data)}.\nResult: {result.data}")

        if not result.data:
            return "No relevant documentation found."

        # Combine the pages
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)

        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        # Join pages
        # return "\n\n---\n\n".join(result.data)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


# @gp_connect_agent.tool
# async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
#     """
#     Retrieve a list of all available GP Connect documentation pages.

#     Returns:
#         List[str]: List of unique URLs for all documentation pages
#     """
#     try:
#         # Query Supabase for unique URLs where source is gpconnect-1-6-0
#         result = (
#             ctx.deps.supabase.from_("site_pages")
#             .select("url")
#             .eq("metadata->>source", "gpconnect-1-6-0")
#             .execute()
#         )

#         if not result.data:
#             return []

#         # Extract unique URLs
#         urls = sorted(set(doc["url"] for doc in result.data))
#         return urls

#     except Exception as e:
#         print(f"Error retrieving documentation pages: {e}")
#         return []


# @gp_connect_agent.tool
# async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
#     """
#     Retrieve the full content of a specific documentation page by combining all its chunks.

#     Args:
#         ctx: The context including the Supabase client
#         url: The URL of the page to retrieve

#     Returns:
#         str: The complete page content with all chunks combined in order
#     """
#     try:
#         # Query Supabase for all chunks of this URL, ordered by chunk_number
#         result = (
#             ctx.deps.supabase.from_("site_pages")
#             .select("title, content, chunk_number")
#             .eq("url", url)
#             .eq("metadata->>source", "gpconnect-1-6-0")
#             .order("chunk_number")
#             .execute()
#         )

#         if not result.data:
#             return f"No content found for URL: {url}"

#         # Format the page with its title and all chunks
#         page_title = result.data[0]["title"].split(" - ")[0]  # Get the main title
#         formatted_content = [f"# {page_title}\n"]

#         # Add each chunk's content
#         for chunk in result.data:
#             formatted_content.append(chunk["content"])

#         # Join everything together
#         return "\n\n".join(formatted_content)

#     except Exception as e:
#         print(f"Error retrieving page content: {e}")
#         return f"Error retrieving page content: {str(e)}"
