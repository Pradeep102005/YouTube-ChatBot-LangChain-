from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

#step1-a INdexing(Document Ingestion)
from youtube_transcript_api import YouTubeTranscriptApi

video_id = "Gfr50f6ZBvo"

try:
    api = YouTubeTranscriptApi()   # create object

    transcript_list = api.fetch(video_id)   # ✅ NEW METHOD
    # print(transcript_list)

    transcript = " ".join(chunk.text for chunk in transcript_list)

    # print(transcript)

except Exception as e:
    print(f"Error: {e}")


#step1-b Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

docs = text_splitter.split_text(transcript)
print(f"Split {len(docs)} documents")

#step 1c& 1d- Embeddings and Vector Store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_texts(docs, embeddings)
print("Vector store created")

#step2 Retrieval
retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
# print(retriever.invoke("What is the impact of AI on the job market?"))

#step3 LLM
llm=ChatOllama(model="gemma2:2b")

#step4 Prompt
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
question= "can you summarize the video?"
retrieved_docs= retriever.invoke(question)

# Helper function to format the documents into plain text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#step5 Chain
chain=prompt|llm|StrOutputParser()

#step6 RAG
final_answer=chain.invoke({
    "context": format_docs(retrieved_docs),
    "question": question
})
print(final_answer)