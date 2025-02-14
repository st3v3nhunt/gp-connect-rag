"""
This script crawls the URLs from the gpconnect-1-6-0 site and stores the
content in the Supabase database.
"""

import asyncio

# import logging
import os

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import tiktoken
from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    CrawlResult,
)
from crawler_lib import get_urls_to_crawl
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

load_dotenv()

# Create OpenAI Supabase clients
openai_client = OpenAI()
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
)


def get_embedding(text: str) -> List[float]:
    """Get embeddings for a given string."""
    response = openai_client.embeddings.create(
        input=text, model="text-embedding-3-small"
    )
    return response.data[0].embedding


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string.
    Takes time to run. Do not use in production."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


@dataclass
class Page:
    """Dataclass to represent a page."""

    url: str
    title: str
    description: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


def insert_page_to_db(page: Page):
    """Insert a page into the database."""
    print(f"insert_page_to_db start for: {page.url}")
    data = {
        "url": page.url,
        "title": page.title,
        "description": page.description,
        "content": page.content,
        "metadata": page.metadata,
        "embedding": page.embedding,
    }
    supabase.table("site_pages").insert(data).execute()
    print(f"insert_page_to_db ended for: {page.url}")


run_cfg = CrawlerRunConfig(
    verbose=True,
    css_selector=".col-md-9",
    cache_mode=CacheMode.BYPASS,  # ensure fresh content
    excluded_tags=["header", "footer"],
)


async def crawl_url(url: str) -> CrawlResult:
    """Crawl the given URL and return the result."""
    print(f"crawl_url start for: {url}")
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
            config=run_cfg,
        )
        print(f"crawl_url ended for: {url}")
        return result


def process_crawl_result(result) -> Page:
    """Process the crawl result and return a Page."""
    print(f"process_crawl_result started for: {result.url}")
    markdown = result.markdown_v2.raw_markdown

    embedding = get_embedding(markdown)
    metadata = {
        "source": "gpconnect-1-6-0",
        "document_size": len(markdown),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"process_crawl_result ended for: {result.url}")
    return Page(
        url=result.url,
        title=result.metadata["title"],
        description=result.metadata["description"],
        content=markdown,
        metadata=metadata,
        embedding=embedding,
    )


async def main():
    """Main function to run the crawler"""
    urls = get_urls_to_crawl()
    for url in urls:
        result = await crawl_url(url)
        page = process_crawl_result(result)
        insert_page_to_db(page)


# Run the async main function
asyncio.run(main())
