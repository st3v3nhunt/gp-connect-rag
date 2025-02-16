# GP Connect RAG

> An agentic RAG system for GP Connect

## Setup

### Environment setup

Activate venv and install dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements
```

Setup the environment variables. Using the `.env.example` file as a template, create a `.env` file and populate the
values of the listed variables.

- `OpenAI` API keys can be obtained from [OpenAI](https://platform.openai.com/api-keys).
- `SUPABASE_*` API keys can be obtained from the project page in
  [Supabase](https://supabase.com/dashboard/project/_/settings/api).
- `PASSWORD` should be set to a strong password/passphrase for when (if) the application is deployed and use of the
  model wants to be retricted.

### Database setup

Create a new project in Supabase. Run the `site_pages.sql` script, it will:

1. Create the table `site_pages`
2. Enable vector similarity search
3. Set up Row Level Security policies

In Supabase, do this by going to the "SQL Editor" tab and pasting in the SQL into the editor there. Then click "Run".

### Crawler, data processing and storage

Run:

```sh
python crawl4ai_crawler.py
```

This will:

1. Crawl the GP Connect documentation
1. Use OpenAI to generate an embedding for each page
1. Store the page content and embedding in Supabase

### Streamlit Web Interface

Run:

```sh
streamlit run streamlit_ui.py
```

This will, create a web interface for the RAG system on [http://localhost:8501](http://localhost:8501).
