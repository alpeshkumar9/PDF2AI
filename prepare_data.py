from pathlib import Path
from typing import Generator
from pdfminer.high_level import extract_text
import unicodedata
import json


class PDFTextExtractor:
    """Class for extracting and processing text from PDF files.

    Attributes:
        pdf_folder (Path): Directory containing PDF files.
        output_file (Path): File path to save the extracted data.
    """

    def __init__(self, pdf_folder: str, output_file: str) -> None:
        """Initialize the PDFTextExtractor with folder and output file paths."""
        self.pdf_folder = Path(pdf_folder)
        self.output_file = Path(output_file)

    @staticmethod
    def replace_ligatures(text: str) -> str:
        """Replace ligatures in the text with their equivalent characters.

        Args:
            text (str): The text to process.

        Returns:
            str: The processed text with ligatures replaced.
        """
        text = unicodedata.normalize('NFKD', text)
        replacements = {
            "\uFB00": "ff", "\uFB01": "fi", "\uFB02": "fl",
            "\uFB03": "ffi", "\uFB04": "ffl", "\uFB05": "ft",
            "\uFB06": "st", "\u2013": "-", "\u2014": "--",
            "\u2018": "'", "\u2019": "'", "\u201C": '"',
            "\u201D": '"', "\u2022": "*", "\u00A0": " ",
            "\u00A3": "GBP ", "\u20AC": "EUR ", "\u2610": "[ ]",
            "\u2611": "[X]", "\u2612": "[X]"
        }
        return ''.join(replacements.get(c, c) for c in text)

    @staticmethod
    def split_into_chunks(text: str, header: str, max_words: int = 750, overlap: int = 50) -> Generator[str, None, None]:
        """Split text into chunks of a specified maximum number of words.

        Args:
            text (str): The text to be split.
            header (str): Header to be added to each chunk.
            max_words (int, optional): Maximum number of words in a chunk. Defaults to 750.
            overlap (int, optional): Number of words to overlap between chunks. Defaults to 50.

        Yields:
            Generator[str, None, None]: Generator of text chunks.
        """
        words = text.split()
        for i in range(0, len(words), max_words):
            start = max(0, i - overlap)
            end = min(i + max_words, len(words))
            yield header + ' '.join(words[start:end])

    def extract_data(self) -> None:
        """Extract text from PDFs in the specified folder and save to the output file."""
        all_messages = []
        for pdf_file in self.pdf_folder.glob('*.pdf'):
            text = extract_text(str(pdf_file))
            text = self.replace_ligatures(text)
            header = f"Filename: {pdf_file.name}\n"
            all_messages.extend({"role": "assistant", "content": chunk}
                                for chunk in self.split_into_chunks(text, header))
        with self.output_file.open('w') as jsonl_file:
            for message in all_messages:
                json.dump({"messages": [message]}, jsonl_file)
                jsonl_file.write('\n')


# Example usage
extractor = PDFTextExtractor('data/train', 'data/train/output.jsonl')
extractor.extract_data()
