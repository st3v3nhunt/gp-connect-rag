from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_list.html"
)

docs = loader.load()

# print(docs[0].page_content[:500])
print(docs[0])
