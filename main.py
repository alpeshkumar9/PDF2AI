import os

import uvicorn
import urllib.parse
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from fastapi import FastAPI, Form, Body
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize FastAPI app and OpenAI client
app = FastAPI()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
chatgpt_model = os.getenv('CHATGPT_MODEL')


@app.get("/", include_in_schema=False)
async def read_root() -> RedirectResponse:
    """
    Redirects to the Swagger UI documentation page.

    Returns:
        RedirectResponse: A redirection to the /docs URL.
    """
    return RedirectResponse(url="/docs")


def extract_text_from_page(pdf_path: str, page_number: int) -> str:
    """
    Extracts text from a specific page of a PDF document.

    Args:
        pdf_path (str): The file path of the PDF document.
        page_number (int): The page number to extract text from.

    Returns:
        str: Extracted text from the specified page.
    """
    pdf_text = ""
    for page_layout in extract_pages(pdf_path):
        if page_layout.pageid == page_number + 1:
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    pdf_text += " ".join(element.get_text().split())

    return pdf_text.strip()


@app.post("/question_answer_by_chagpt/", tags=["ChatGPT"])
async def question_answer_by_chagpt(
    page_number: int = Form(...),
    question: str = Body(...)
) -> dict:
    """
    Generates an answer to a given question based on the text extracted from a specified page of a PDF document.

    Args:
        page_number (int): The page number of the PDF document to reference.
        question (str): The question to be answered.

    Returns:
        dict: A dictionary containing the question and the generated answer.
    """
    pdf_document_path = "data/Attention_Is_All You_Need.pdf"
    text = extract_text_from_page(pdf_document_path, page_number=page_number)

    prompt = f"Document: {text}\n\nQuestion: {question}\nAnswer:"
    response = client.completions.create(
        model=chatgpt_model,
        prompt=prompt,
        max_tokens=500,
        temperature=0.2,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=1,
        stop='\n',
        logprobs=0,
        echo=False
    )

    answer = response.choices[0].text.strip()
    answer = answer.replace("answer=", "")
    return {'question': question, 'answer': urllib.parse.unquote_plus(answer)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
