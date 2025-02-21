# Notes

## Content filtering

The HTML source could be better processed and specifically constrained to ensure only the most relevant and useful content
is converted into markdown. Using Crawl4AI this could be done by an implementation of RelevantContentFilter. At time of
writing, neither the pruning or BM25 content filtering techniques are suitable to constrain the content.

## Sand

Need to take the question in from the user.
Get the embedding of the question.
Compare the embedding of the question to the embeddings in the DB.
Return the content.

## Questions

- Why is the request to embedding changing the user query? Is it possible to prevent that from happening?
  - The user_query is being changed by the LLM before is makes the request. Change the prompt doesn't seem to make a
    difference.
