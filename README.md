# PDF2AI

## Overview

This application integrates a PDF text extraction feature and store as embedding vector in vector database (FAISS) and use OpenAI's ChatGPT model to provide an interactive question-answering system. It allows users to query a PDF document and receive contextually relevant answers from vector database which is converted into more meaningful answer by ChatGPT.

## Features

- PDF Text Extraction: Extract text from any given page of a PDF document.
- Embedding Vector: Store the extracted text into embedding vector form in a vector datbase.
- ChatGPT-Powered Responses: Generate answers to questions based on the content of a specified page in the PDF document.
- Web-Based Interface: The application is accessible through a web interface, allowing for easy interaction and use.

## How It Works

1. PDF Selection and Page Reference: Users can specify a page number from a pre-defined PDF document.
2. Question Input: Users can input a question related to the content of the selected page.
3. Similarity Search: Answers which are similar to the asked questions is search from the vector database.
4. Answer Generation: The application processes the extracted text from the PDF page and the user's question, leveraging ChatGPT to generate a relevant answer.
5. Response Display: The generated answer is displayed to the user, providing insights or information based on the PDF's content.

## Technology

- FastAPI: Powers the backend of the application, handling web requests and server-side logic.
- PDFMiner: Used for extracting text from PDF documents.
- FAISS: Vector database to store the embedding vectors and perform similarity search.
- OpenAI's ChatGPT: Provides the AI model for generating answers to user queries.
- Uvicorn: Serves as the ASGI server for hosting the application.

## Setup and Usage

### Prerequisites

Before you start, ensure you have the following installed:

- Python 3.6 or higher
- Pip (Python package installer)

#### Installation

1. Clone the Repository:
   If the application is hosted in a Git repository, provide instructions to clone it. Otherwise, skip this step if the user is setting it up directly from provided files.

```
git clone [your-repository-link]
cd [repository-name]
```

2. Environment Setup:
   It's recommended to use a virtual environment for Python projects. This keeps dependencies required by different projects separate and organized.

```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install Dependencies:
   Install the required Python packages using pip.

```
pip install -r requirements.txt
```

4. Environment Variables:
   Set up the necessary environment variables. Create a .env file in the root directory of the project and add the following variables:

```
OPENAI_API_KEY=your_openai_api_key
CHATGPT_MODEL=model_name  # for example, "davinci"
```

Replace your_openai_api_key and model_name with your actual OpenAI API key and the model name you intend to use.

#### Running the Application

1. Start the API Server:
   Run the following command to start the FastAPI server:
   ```
   python main.py
   ```
2. Start the streamlit server in new terminal tab.
   ```
   streamlit run app.py
   ```
   It will open application in browser at http://localhost:8501/

#### Interacting with the Application

- Through the frontend, you can test the PDF text extraction and question-answering features.
- Select a page number and input your question related to the content on that page.
- Submit the request, and the application will display the generated answer based on the PDF's content.

### Notes

- Ensure that the PDF file (data/Attention_Is_All You_Need.pdf) is placed in the correct directory as specified in the code.
- The API key and model name must be valid and active for the OpenAI service to work correctly.

## Demo

![Screenshot 2023-12-06 at 1 05 23 pm](https://github.com/alpeshkumar9/PDF2AI/assets/8064993/a5dac824-762f-4fe2-ad0f-dac61d313ff4)

## Application Flow

![Screenshot 2023-11-13 at 11 42 40 am](https://github.com/alpeshkumar9/PDF2AI/assets/8064993/c8024781-1671-4768-927c-d835b9837850)
