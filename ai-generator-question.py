import json
from pathlib import Path
from google import genai

API_KEY = "AIzaSyAtNIe1Xnv31PvM3DbWd_NwRnsP9DMw-oo"
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"

OUTPUT_FOLDER = "output"
EXAM_SPEC_FILE = "exam_spec_example.json"
EXAM_OUTPUT_FILE = "example_exam.json"

def build_exam_prompt(pdf_datas, exam_spec):
    prompt = "Generate an exam based on the following content from multiple documents.\n\n"

    for pdf_data in pdf_datas:
        prompt += f"Document Title: {pdf_data.get('title', '')}\n"
        for section in pdf_data.get("sections", []):
            prompt += f"Section: {section.get('section_title', '')}\n"
            prompt += f"Content: {section.get('section_text', '')}\n\n"

    prompt += "Exam Specification:\n"
    prompt += f"- Number of MCQ questions: {exam_spec.get('req_mcq', 5)}\n"
    prompt += f"- Number of True/False questions: {exam_spec.get('req_tf', 5)}\n"
    prompt += f"- Number of Essay questions: {exam_spec.get('req_essay', 2)}\n"

    difficulty = exam_spec.get("difficulty_percentages", {})
    prompt += "- Difficulty distribution:\n"
    prompt += f"  * Easy: {difficulty.get('easy', 0)}%\n"
    prompt += f"  * Medium: {difficulty.get('medium', 0)}%\n"
    prompt += f"  * Hard: {difficulty.get('hard', 0)}%\n"

    bloom_levels = exam_spec.get("bloom_levels", [])
    if bloom_levels:
        prompt += "- Bloom levels to use (randomly distributed among questions): " + ", ".join(bloom_levels) + "\n"

    topics = exam_spec.get("topics", [])
    if topics:
        prompt += "- Topics to include:\n"
        for t in topics:
            prompt += f"  * {t}\n"

    prompt += "\nPlease provide the exam in a clean JSON format with questions and answers clearly structured, matching the specification above."
    return prompt

def main():
   
    with open(EXAM_SPEC_FILE, "r", encoding="utf-8") as f:
        exam_spec = json.load(f)

    pdf_files = [f for f in Path(OUTPUT_FOLDER).glob("*.json")]

    pdf_datas = []
    for file in pdf_files:
        with open(file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                pdf_datas.append(data)
            except json.JSONDecodeError:
                print(f"⚠️ Failed to read {file}, skipping.")

    if not pdf_datas:
        print("❌ No valid PDF JSON files found in output folder.")
        return
    prompt = build_exam_prompt(pdf_datas, exam_spec)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    with open(EXAM_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"✅ Exam generated and saved as {EXAM_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
