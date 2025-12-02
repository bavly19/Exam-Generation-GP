import os
import json
from pathlib import Path

OUTPUT_DIR = "output"
SPEC_FILE = "exam_spec_example.json"

BLOOM_OPTIONS = [
    "Knowledge",
    "Comprehension",
    "Application",
    "Analysis",
    "Synthesis",
    "Evaluation"
]

def get_all_topics_from_outputs(output_dir):
    topics = set()
    for json_file in Path(output_dir).glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "sections" in data:
                    for section in data["sections"]:
                        if "section_title" in section:
                            topics.add(section["section_title"])
        except Exception as e:
            print(f"⚠️ Warning: failed to read {json_file}: {e}")
    return list(topics)

def prompt_user_for_exam_spec(topics):
    print("Found the following topics in all output files:")
    for i, topic in enumerate(topics, start=1):
        print(f"{i}. {topic}")
    
    print("\nEnter the numbers of the topics you want to include, separated by commas, or 'all' for all topics:")
    topic_input = input("Topics: ").strip()
    if topic_input.lower() == "all":
        selected_topics = topics
    else:
        try:
            indices = [int(x.strip()) - 1 for x in topic_input.split(",")]
            selected_topics = [topics[i] for i in indices if 0 <= i < len(topics)]
        except:
            print("Invalid input, using all topics.")
            selected_topics = topics

    # Number of questions
    try:
        req_mcq = int(input("Number of MCQ questions: ").strip())
    except:
        req_mcq = 5
    try:
        req_tf = int(input("Number of True/False questions: ").strip())
    except:
        req_tf = 5
    try:
        req_essay = int(input("Number of Essay questions: ").strip())
    except:
        req_essay = 2
    print("\nEnter the percentage for each difficulty level (sum should not exceed 100):")
    while True:
        try:
            easy_pct = int(input("Easy %: ").strip())
            medium_pct = int(input("Medium %: ").strip())
            hard_pct = int(input("Hard %: ").strip())
            total = easy_pct + medium_pct + hard_pct
            if total > 100:
                print(f"⚠️ Total percentage is {total}%, which exceeds 100%. Please enter again.")
                continue
            break
        except:
            print("Invalid input. Please enter integer values.")
    print("\nBloom taxonomy options (applied to all questions in the exam):")
    for i, bloom in enumerate(BLOOM_OPTIONS, start=1):
        print(f"{i}. {bloom}")

    print("\nSelect Bloom levels to include (numbers separated by commas, or 'all'):")
    bloom_input = input("Bloom levels: ").strip()
    if bloom_input.lower() == "all":
        selected_bloom = BLOOM_OPTIONS
    else:
        try:
            indices = [int(x.strip()) - 1 for x in bloom_input.split(",")]
            selected_bloom = [BLOOM_OPTIONS[i] for i in indices if 0 <= i < len(BLOOM_OPTIONS)]
        except:
            selected_bloom = BLOOM_OPTIONS

    exam_spec = {
        "req_mcq": req_mcq,
        "req_tf": req_tf,
        "req_essay": req_essay,
        "difficulty_percentages": {
            "easy": easy_pct,
            "medium": medium_pct,
            "hard": hard_pct
        },
        "bloom_levels": selected_bloom,
        "topics": selected_topics
    }
    return exam_spec

def main():
    topics = get_all_topics_from_outputs(OUTPUT_DIR)
    if not topics:
        print("No topics found in output files.")
        return
    exam_spec = prompt_user_for_exam_spec(topics)
    
    with open(SPEC_FILE, "w", encoding="utf-8") as f:
        json.dump(exam_spec, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ Exam spec saved to {SPEC_FILE}")

if __name__ == "__main__":
    main()
