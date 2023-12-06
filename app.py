import io
import os

from dotenv import load_dotenv
import fitz  # PyMuPDF
import streamlit as st
import requests
from PIL import Image

# Function to display PDF as images

load_dotenv()
API_URL = os.getenv('API_URL')


def display_pdf(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    image = page.get_pixmap()
    img = Image.frombytes(
        "RGB", [image.width, image.height], image.samples)
    img_stream = io.BytesIO()
    img.save(img_stream, format='PNG')
    img_stream.seek(0)  # Rewind to the beginning of the stream
    st.image(
        img_stream, caption=f"Page {page_number + 1}", use_column_width=True)


# Set page layout to wide
st.set_page_config(layout="wide")

# Initialize session state for messages and selected page number
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_page_number" not in st.session_state:
    st.session_state.selected_page_number = 0

# Handle chat input
if prompt := st.chat_input("Ask a Question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = requests.post(
        f"{API_URL}/question_answer_by_chagpt/",
        data={"question": prompt,
              "page_number": st.session_state.selected_page_number}
    )
    if response.status_code == 200:
        answer = response.json()['answer']
    else:
        answer = f"Error: {response.status_code}"
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Main container
with st.container():
    col1, col2 = st.columns(2)

    # Chat interface
    with col1:
        st.header("Chat")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # PDF preview
    with col2:
        st.header("PDF Preview")
        file = 'data/Attention_Is_All You_Need.pdf'
        doc = fitz.open(file)
        total_pages = len(doc)

        # Page selection
        page_number = st.selectbox(
            'Select Page Number:',
            options=list(range(1, total_pages + 1)),
            index=st.session_state.selected_page_number
        )
        st.session_state.selected_page_number = page_number - 1

        # Display PDF
        display_pdf(file, st.session_state.selected_page_number)
        st.caption(
            "Source: [https://doi.org/10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)")
