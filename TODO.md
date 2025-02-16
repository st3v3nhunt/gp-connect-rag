# To do

## Core

- [X] Get list of pages to scrape
- [X] Scrape the pages and process data
  - [X] Reduce page data to main content
  - [X] Generate the embedding
- [X] Load it into Supabase
- [X] Create Q&A/RAG agent
- [X] Create UI, using Streamlit

## Future work

- [ ] Deploy to Streamlit cloud
- [ ] Include links to the page in message response to user
- [ ] Add logging
- [ ] Handle errors, general non-happy path stuff
  - [ ] Context length of content is too large to create embedding for
    [FHIR implementation](https://developer.nhs.uk/apis/gpconnect-1-6-0/development_fhir_api_guidance.html) and
    [FHIR investigations examples](https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_fhir_examples_pathology.html) and
    [Retrieve a patient's structured record](https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_retrieve_patient_record.html)
- [ ] Dynamically generate the list of pages to scrape
- [ ] Add content for Access Document
- [ ] Add content for Access Record HTML
- [ ] Improve data processing and cleaning, look at BS4 and Langchain options
- [ ] Look into Langchain for scraping the page and processing the data
- [ ] Add timing information to the processing
- [ ] Allow for content being updated
- [ ] Generate a description based on the content rather than using the one from the page as they are all the same
- [ ] Add progress information to the crawl e.g. X out of Y URLs have been processed
- [ ] Try a different model e.g. Gemini Flash 2.0
