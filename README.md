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

Populate env vars. Using the `.env.example` file as a template, create a `.env` file and populate the values of the
listed variables.

### Database setup

Create a new project in Supabase. Run the `site_pages.sql` script, it will:

1. Create the table `site_pages`
2. Enable vector similarity search
3. Set up Row Level Security policies

In Supabase, do this by going to the "SQL Editor" tab and pasting in the SQL into the editor there. Then click "Run".

### Crawler

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

This will, create a web interface for the RAG system on `http://localhost:8501`.
