# Notes

## Content filtering

The HTML source could be better processed and specifically constrained to ensure only the most relevant and useful content
is converted into markdown. Using Crawl4AI this could be done by an implementation of RelevantContentFilter. At time of
writing, neither the pruning or BM25 content filtering techniques are suitable to constrain the content.

## Notices

- Why is the request to embedding changing the user query? Is it possible to prevent that from happening?
  - The user_query is being changed by the LLM before is makes the request. Changing the prompt doesn't make a
    difference. It is possible to use the raw user entered text via `ctx.prompt` but this would not be a good idea.
    It removes some of the benefit of having the LLM create the question.
- Embeddings are case sensitive. Asking the LLM (at least OpenAI 4 mini) to use the 'normal' capitalisation doesn't
  appear to make any difference.
