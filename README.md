# Smart Q&A Application with OpenAI and Pinecone Integration
## A "retrieval augmented generation" (RAG) app with Langchain and OpenAI in Python + Gradio interface + Pinecone vector database.

The "Smart Q&A Application with OpenAI and Pinecone Integration" is a simple Python application designed for question-answering tasks. Leveraging powerful technologies such as OpenAI for natural language understanding and `Pinecone` for efficient similarity search, this application offers a range of features to enhance the user's experience:

1. **Document Processing:**
   - The app allows users to input a directory of documents for analysis.
   - Documents are efficiently processed and broken down into smaller, manageable chunks.

2. **Embeddings Generation with OpenAI:**
   - Utilizes the `OpenAI API` to generate embeddings from `LLM`s for the processed document chunks.
   - Embeddings capture semantic information, enabling better understanding of the content.
   - Leverage "retrieval augmented generation" **(RAG)** from `Langchain`.

3. **Efficient Search with Pinecone:**
   - Establishes a connection to the Pinecone service for efficient similarity search.
   - Creates an index to store and retrieve document embeddings.

4. **OpenAI Language Model Integration:**
   - Incorporates OpenAI's powerful language model for advanced natural language processing.
   - Fine-tuned parameters, such as temperature, enhance the quality of responses.

5. **Question-Answer Chain:**
   - Implements a question-answer chain from the OpenAI language model, enabling a dynamic and contextualized Q&A experience.

6. **User-Friendly Gradio Interface:**
   - The application features an interactive user interface created with `Gradio`.
   - Users can input their queries using a textbox, enhancing user interaction and accessibility.

7. **Real-Time Results Display:**
   - Results are displayed in real-time in another textbox, providing users with quick and relevant answers to their queries.

8. **Extensibility and Customization:**
   - The application can be easily extended and customized to fit different use cases, making it a versatile tool for various domains.
   - Easily extendable to display source urls.

9. **Smart Search Inside PDFs:**
   - The app includes an informative user prompt, encouraging users to search inside `PDF`s, suggesting its capability to handle PDF documents.

Overall, this application amalgamates cutting-edge technologies to create an intelligent Q&A system, making it a valuable tool for tasks that require natural language understanding and efficient document retrieval.

## How to use:
1. Install requirements:
```python
pip install -r requirements.txt
```
2. Place your Environment variables in the `.env` file.
3. Run the app:
```python
python app.py
```
4. Visit http://127.0.0.1:7860/ on your browser.
