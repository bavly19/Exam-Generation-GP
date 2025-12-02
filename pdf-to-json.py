import os
import json
import re
import logging
import argparse
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import time
from tqdm import tqdm

from pypdf import PdfReader
from google import genai
from google.api_core import exceptions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    api_key: str = "AIzaSyAtNIe1Xnv31PvM3DbWd_NwRnsP9DMw-oo"
    model: str = "gemini-2.5-flash"
    max_retries: int = 3
    retry_delay: float = 1.0
    output_dir: str = "output"
    temp_dir: str = "temp"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

class PDFProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.api_key)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            logger.info(f"Starting text extraction from file: {pdf_path}")
            reader = PdfReader(pdf_path)
            text = ""
            for page_num, page in enumerate(tqdm(reader.pages, desc="Processing pages"), start=1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        page_text = self._clean_page_text(page_text)
                        page_text += self._extract_images_equations(page)
                        text += f"\n\n--- Page {page_num} ---\n\n" + page_text
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                    continue
            logger.info(f"Successfully extracted text from {len(reader.pages)} pages")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def _extract_images_equations(self, page) -> str:
        content_text = ""
        try:
            if hasattr(page, "images") and page.images:
                for idx, img in enumerate(page.images, 1):
                    content_text += f"[Image {idx} detected]\n"

            if hasattr(page, "tables") and page.tables:
                for idx, tbl in enumerate(page.tables, 1):
                    content_text += f"[Table {idx} detected]\n"

            content_text += "[Detected equations will be described]\n"
        except Exception as e:
            logger.debug(f"No images/tables/equations detected on page: {str(e)}")
        return content_text


    def _clean_page_text(self, text: str) -> str:
        text = re.sub(r'\s*\[.*?(image|figure|table).*?\]\s*', '', text, flags=re.I)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'Page \d+ of \d+', '', text)
        return text.strip()

    def call_gemini_api(self, prompt: str) -> str:
        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Sending request to Gemini API (attempt {attempt + 1}/{self.config.max_retries})")
                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=prompt
                )
                return response.text
            except exceptions.GoogleAPICallError as e:
                last_exception = e
                logger.warning(f"API call error: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error: {str(e)}")
                break
        logger.error("All connection attempts to Gemini API failed")
        raise last_exception if last_exception else Exception("Failed to connect to Gemini API")

    def structure_text_with_gemini(self, text: str) -> Dict[str, Any]:
        prompt = f"""
Clean and structure the following text into a JSON with these requirements:
1. Fields:
   - "title": string, the main title of the document.
   - "sections": list of objects, each with:
       * "section_title": string
       * "section_text": string
2. Detect and describe images, figures, tables, and mathematical equations in the text.
3. Automatically detect and separate content into clear topics/sections.
4. Ensure the JSON is well-formatted, readable, and ready for further processing.
5. Return ONLY valid JSON without any additional text or explanations.

CONTENT:
{text}
"""
        try:
            response = self.call_gemini_api(prompt)
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    logger.warning("Could not parse Gemini response as JSON, saving raw text instead.")
                    return {"raw_text": response, "error": "Failed to parse as JSON"}
        except Exception as e:
            logger.error(f"Error structuring text with Gemini: {str(e)}")
            return {"error": str(e), "raw_text": text}

    def pdf_to_json(self, pdf_path: str) -> str:
        pdf_name = Path(pdf_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{pdf_name}_{timestamp}.json"
        output_path = os.path.join(self.config.output_dir, output_filename)

        try:
            pdf_text = self.extract_text_from_pdf(pdf_path)

            temp_text_path = os.path.join(self.config.temp_dir, f"{pdf_name}_raw.txt")
            with open(temp_text_path, "w", encoding="utf-8") as f:
                f.write(pdf_text)
            logger.info(f"Saved raw text to: {temp_text_path}")

            structured_data = self.structure_text_with_gemini(pdf_text)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=4)

            logger.info(f"Saved JSON to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error converting PDF to JSON: {str(e)}")
            raise


def main():
    config = Config()

    if not config.api_key:
        logger.error("Gemini API key is required. Set GEMINI_API_KEY environment variable.")
        return

    list_file = "list-of-pdfs.json"
    if not os.path.exists(list_file):
        logger.error(f"List of PDFs not found: {list_file}")
        return

    with open(list_file, "r", encoding="utf-8") as f:
        try:
            pdf_list_data = json.load(f)
            pdf_files = pdf_list_data.get("pdf_files", [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to read JSON list file: {str(e)}")
            return

    processor = PDFProcessor(config)

    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            logger.warning(f"File does not exist, skipping: {pdf_path}")
            continue
        try:
            output_json = processor.pdf_to_json(pdf_path)
            print(f"✅ JSON saved to: {output_json}")
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
