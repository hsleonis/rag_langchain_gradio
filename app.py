from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from utils import read_doc, chunk_data, retrieve_answers, qa_manager
import gradio as gr
from environment import OPENAI_API_KEY, \
    PINECONE_API_KEY, PINECONE_ENVIRONMENT, \
    PINECONE_INDEX, DOC_DIR_PATH, OPENAI_LLM


def main(query: str):
    # Read docs
    doc = read_doc(DOC_DIR_PATH)

    # Split docs to smaller chunks
    documents = chunk_data(doc)

    # Get OpenAI embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Connect Pinecone client
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT
    )

    # Database index
    index_name = PINECONE_INDEX

    # Store embeddings in Pinecone
    index = Pinecone.from_documents(doc, embeddings, index_name=index_name)

    # Load LLM from OpenAI
    llm = OpenAI(model_name=OPENAI_LLM, temperature=0.3)

    # Load Question-Answer Chain from LLM
    chain = load_qa_chain(llm, chain_type="stuff")

    return retrieve_answers(index, chain, query)


if __name__ == '__main__':
    qa_app = gr.Interface(
        fn=qa_manager,
        inputs=[gr.Textbox(label="What are you looking for?", info="Search inside PDFs.")],
        outputs=[gr.Textbox(label="Results")],
        title="Smart Q&A Application with OpenAI and Pinecone Integration",
        description="A 'retrieval augmented generation' (RAG) app with Langchain and OpenAI in"
                    " Python + Gradio interface + Pinecone vector database."
    )

    # Launch Q&A app
    qa_app.launch()
