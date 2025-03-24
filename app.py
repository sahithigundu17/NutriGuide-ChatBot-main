import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils import HuggingFaceLLM
from PIL import Image
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import camelot
import pytesseract
from pdf2image import convert_from_path

# Initialize session state
def initialize_session_state():
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('generated', ["Hello! I am here to provide answers to questions extracted from uploaded PDF files."])
    st.session_state.setdefault('past', ["Hello Buddy!"])

# Handle conversation
def conversation_chat(user_input, history, llm):
    response_text = llm._call(user_input)
    history.append((user_input, response_text))
    return response_text

# Generate response
def generate_response(user_input, llm):
    with st.spinner('Spinning a snazzy reply...'):
        output = conversation_chat(user_input, st.session_state['history'], llm)
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

# Display chat UI
def display_generated_responses(reply_container):
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="adventurer")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

def display_chat(llm):
    reply_container = st.container()
    container = st.container()
    with container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me questions from uploaded PDF", key='input')
            submit_button = st.form_submit_button(label='Send ⬆️')
        if submit_button and user_input:
            generate_response(user_input, llm)
    display_generated_responses(reply_container)

# Handle PDF upload with OCR support
def handle_pdf_upload():
    pdf_files = st.sidebar.file_uploader("Upload PDF", accept_multiple_files=True, type="pdf")
    if pdf_files:
        text = []
        tables = []
        for file in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            try:
                loader = PyPDFLoader(temp_file_path)
                text_docs = loader.load()
                if text_docs:
                    text.extend(text_docs)
                else:
                    st.warning("No text found in PDF! Trying OCR...")
                    images = convert_from_path(temp_file_path)
                    for img in images:
                        extracted_text = pytesseract.image_to_string(img)
                        text.append(extracted_text)
                try:
                    tables_from_pdf = camelot.read_pdf(temp_file_path, pages='all', flavor='stream')
                    if tables_from_pdf and tables_from_pdf.n > 0:
                        for table in tables_from_pdf:
                            tables.append(table.df)
                except:
                    st.warning("Could not extract tables. The PDF might be scanned.")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=10)
        text_chunks = text_splitter.split_documents(text)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        if tables:
            st.write("Tables extracted from the PDF:")
            for idx, table in enumerate(tables):
                st.write(f"Table {idx + 1}:")
                st.dataframe(table)
        return vector_store
    else:
        return None

def main():
    initialize_session_state()
    st.title("NUTRIGUIDE")
    image = Image.open('Image.jpeg')
    st.image(image, width=500)
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    vector_store = handle_pdf_upload()
    if vector_store:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = HuggingFaceLLM()
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=vector_store.as_retriever(search_kwargs={"k": 2}), 
            memory=memory
        )
        display_chat(llm)
    else:
        st.write("No PDF files uploaded yet!")

if __name__ == "__main__":
    main()
