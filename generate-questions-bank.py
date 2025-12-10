#!/usr/bin/env python3
"""
generate-questions-bank.py

- Designed to work with local SLM (Qwen2.5-7B-Instruct via Ollama) if available.
- If Ollama/model not available, falls back to local template-based generator.
- Produces 3 banks (easy/medium/hard) with questions split per Bloom taxonomy:
  Knowledge, Comprehension, Application, Analysis, Synthesis, Evaluation
- For each Topic x Difficulty x Bloom level:
  - 2 MCQs (4 options + correct answer)
  - 2 True/False
  - 2 Essay prompts (with model answer)
- Generates university-level academic questions suitable for exams
"""

import os
import json
import random
import re
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# ----------------------
# Config
# ----------------------
OUTPUT_DIR = "output"
BANKS_DIR = "banks"
COMBINED_OUTPUT = "question_bank.json"

# per-topic counts (per bloom level)
MCQ_PER_BLOOM = 2
TF_PER_BLOOM = 2
ESSAY_PER_BLOOM = 2

DIFFICULTIES = ["easy", "medium", "hard"]
BLOOM_LEVELS = ["Knowledge", "Comprehension", "Application", "Analysis", "Synthesis", "Evaluation"]

# model config (local)
DEFAULT_MODEL = "qwen2.5-7b-instruct"  # change if you named the model differently in your Ollama repo
OLLAMA_CMD = "ollama"  # must be in PATH

random.seed(42)

# ----------------------
# Text utilities (used for fallback generator)
# ----------------------
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ").strip()
    sents = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sents or [text.strip()]

def first_n_sentences(text: str, n: int = 2) -> str:
    sents = split_sentences(text)
    return " ".join(sents[:n]).strip()

def normalize_space(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def extract_keywords(text: str, max_k: int = 5) -> List[str]:
    text = re.sub(r'[^A-Za-z0-9\s\-]', ' ', text)
    words = [w for w in text.split() if len(w) > 3]
    caps = re.findall(r'\b([A-Z][a-z0-9]{3,}(?:\s+[A-Z][a-z0-9]{3,})*)\b', text)
    kws = []
    for c in caps:
        if c not in kws:
            kws.append(c)
    freq = {}
    for w in words:
        lw = w.lower()
        freq[lw] = freq.get(lw, 0) + 1
    for w, _ in sorted(freq.items(), key=lambda x: -x[1]):
        if w.capitalize() not in kws:
            kws.append(w.capitalize())
        if len(kws) >= max_k:
            break
    return kws[:max_k] if kws else ["Concept"]

def extract_concepts(text: str) -> List[str]:
    """Extract key concepts from text for question generation"""
    # Look for definitions, important terms, and relationships
    concepts = []
    
    # Find sentences with definition patterns
    def_patterns = [
        r'(\w+(?:\s+\w+)*)\s+(?:is|are|refers to|can be defined as)\s+([^.]*)',
        r'(\w+(?:\s+\w+)*)\s+(?:means|represents)\s+([^.]*)',
        r'([^.]*)\s+(?:is|are)\s+(\w+(?:\s+\w+)*)'
    ]
    
    for pattern in def_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 2:
                concepts.extend([match[0], match[1]])
    
    # Find important technical terms (capitalized words)
    caps = re.findall(r'\b([A-Z][a-z0-9]{3,}(?:\s+[A-Z][a-z0-9]{3,})*)\b', text)
    concepts.extend(caps)
    
    # Remove duplicates and short terms
    concepts = [c for c in concepts if len(c) > 3]
    concepts = list(set(concepts))
    
    return concepts[:10] if concepts else ["Concept"]

def extract_relationships(text: str) -> List[str]:
    """Extract relationships between concepts for question generation"""
    relationships = []
    
    # Look for relationship patterns
    rel_patterns = [
        r'(\w+(?:\s+\w+)*)\s+(?:depends on|relies on|requires)\s+(\w+(?:\s+\w+)*)',
        r'(\w+(?:\s+\w+)*)\s+(?:affects|influences|impacts)\s+(\w+(?:\s+\w+)*)',
        r'(\w+(?:\s+\w+)*)\s+(?:improves|enhances|optimizes)\s+(\w+(?:\s+\w+)*)',
        r'(\w+(?:\s+\w+)*)\s+(?:reduces|decreases|minimizes)\s+(\w+(?:\s+\w+)*)'
    ]
    
    for pattern in rel_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 2:
                relationships.append(f"{match[0]} - {match[1]}")
    
    return relationships[:5] if relationships else ["Concept1 - Concept2"]

# ----------------------
# Fallback (local) question builders
# ----------------------
def make_mcq_local(topic: str, content: str, difficulty: str, bloom: str, variant: int) -> Dict:
    concepts = extract_concepts(content)
    relationships = extract_relationships(content)
    sents = split_sentences(content)
    
    # Choose question type based on Bloom level
    if bloom == "Knowledge":
        # Definition or identification question
        if concepts and random.random() < 0.7:
            concept = concepts[variant % len(concepts)]
            question = f"Which of the following best describes {concept}?"
            
            # Create plausible options
            correct = f"The correct definition of {concept}"
            distractors = [
                f"A related but incorrect definition of {concept}",
                f"An unrelated concept",
                f"A partially correct but incomplete definition"
            ]
        else:
            # Fallback to a generic question
            question = f"What is the primary focus of {topic}?"
            correct = f"The correct answer about {topic}"
            distractors = [
                f"A related but incorrect aspect of {topic}",
                f"An unrelated concept",
                f"A partially correct but incomplete answer"
            ]
            
    elif bloom == "Comprehension":
        # Explanation or comparison question
        if relationships:
            rel = relationships[variant % len(relationships)]
            parts = rel.split(" - ")
            if len(parts) == 2:
                question = f"How does {parts[0]} relate to {parts[1]}?"
                correct = f"The correct relationship between {parts[0]} and {parts[1]}"
                distractors = [
                    f"An incorrect relationship between {parts[0]} and {parts[1]}",
                    f"A relationship between {parts[0]} and an unrelated concept",
                    f"A relationship between an unrelated concept and {parts[1]}"
                ]
            else:
                question = f"What is the main principle behind {topic}?"
                correct = f"The correct principle behind {topic}"
                distractors = [
                    f"A related but incorrect principle",
                    f"An unrelated principle",
                    f"A partially correct but incomplete principle"
                ]
        else:
            question = f"What is the main principle behind {topic}?"
            correct = f"The correct principle behind {topic}"
            distractors = [
                f"A related but incorrect principle",
                f"An unrelated principle",
                f"A partially correct but incomplete principle"
            ]
            
    elif bloom == "Application":
        # Application or scenario-based question
        question = f"In a practical scenario, how would {topic} be applied?"
        correct = f"The correct application of {topic}"
        distractors = [
            f"An incorrect application of {topic}",
            f"An application of a related but incorrect concept",
            f"A partially correct but incomplete application"
        ]
        
    elif bloom == "Analysis":
        # Analysis or breakdown question
        question = f"What are the key components of {topic}?"
        correct = f"The correct components of {topic}"
        distractors = [
            f"An incorrect set of components for {topic}",
            f"Components of a related but incorrect concept",
            f"A partially correct but incomplete set of components"
        ]
        
    elif bloom == "Synthesis":
        # Synthesis or combination question
        question = f"How could {topic} be integrated with other concepts?"
        correct = f"The correct integration of {topic} with other concepts"
        distractors = [
            f"An incorrect integration of {topic} with other concepts",
            f"An integration of a related but incorrect concept",
            f"A partially correct but incomplete integration"
        ]
        
    else:  # Evaluation
        # Evaluation or judgment question
        question = f"What are the advantages and disadvantages of {topic}?"
        correct = f"The correct advantages and disadvantages of {topic}"
        distractors = [
            f"An incorrect set of advantages and disadvantages for {topic}",
            f"Advantages and disadvantages of a related but incorrect concept",
            f"A partially correct but incomplete set of advantages and disadvantages"
        ]
    
    # Adjust difficulty
    if difficulty == "easy":
        # Make options more distinct
        distractors = [d + " (clearly incorrect)" for d in distractors]
    elif difficulty == "hard":
        # Make options more similar
        distractors = [d.replace("incorrect", "potentially correct") for d in distractors]
    
    # Create final options
    options = [correct] + distractors
    random.shuffle(options)
    correct_index = options.index(correct)
    
    return {
        "question": question,
        "options": options,
        "answer": options[correct_index],
        "difficulty": difficulty,
        "bloom_level": bloom,
        "topic": topic  # Keep topic for internal reference but not shown in question
    }

def make_tf_local(topic: str, content: str, difficulty: str, bloom: str, variant: int) -> Dict:
    concepts = extract_concepts(content)
    relationships = extract_relationships(content)
    sents = split_sentences(content)
    
    # Choose question type based on Bloom level
    if bloom == "Knowledge":
        # Fact-based statement
        if concepts:
            concept = concepts[variant % len(concepts)]
            statement = f"{concept} is a key component of {topic}."
        else:
            statement = f"{topic} is an important concept in this field."
            
    elif bloom == "Comprehension":
        # Relationship-based statement
        if relationships:
            rel = relationships[variant % len(relationships)]
            parts = rel.split(" - ")
            if len(parts) == 2:
                statement = f"{parts[0]} is directly related to {parts[1]}."
            else:
                statement = f"The principles of {topic} are well-established."
        else:
            statement = f"The principles of {topic} are well-established."
            
    elif bloom == "Application":
        # Application-based statement
        statement = f"{topic} can be effectively applied in practical scenarios."
        
    elif bloom == "Analysis":
        # Analysis-based statement
        statement = f"The components of {topic} work together in a predictable manner."
        
    elif bloom == "Synthesis":
        # Synthesis-based statement
        statement = f"{topic} can be effectively integrated with other concepts."
        
    else:  # Evaluation
        # Evaluation-based statement
        statement = f"The benefits of {topic} outweigh its limitations."
    
    # Adjust truth value based on difficulty
    if difficulty == "easy":
        # Easy questions are more likely to be true
        is_true = random.random() < 0.8
    elif difficulty == "medium":
        # Medium questions have equal chance
        is_true = random.random() < 0.5
    else:  # hard
        # Hard questions are more likely to be false
        is_true = random.random() < 0.3
    
    # If we need to make the statement false, modify it
    if not is_true:
        if re.search(r'\b(is|are)\b', statement):
            statement = re.sub(r'\b(is|are)\b', lambda m: 'is not' if m.group(1) == 'is' else 'are not', statement)
        elif re.search(r'\b(can|could)\b', statement):
            statement = re.sub(r'\b(can|could)\b', lambda m: 'cannot' if m.group(1) == 'can' else 'could not', statement)
        else:
            statement = "It is false that " + statement[0].lower() + statement[1:]
    
    return {
        "question": statement,
        "answer": bool(is_true),
        "difficulty": difficulty,
        "bloom_level": bloom,
        "topic": topic  # Keep topic for internal reference but not shown in question
    }

def make_essay_local(topic: str, content: str, difficulty: str, bloom: str, variant: int) -> Dict:
    concepts = extract_concepts(content)
    relationships = extract_relationships(content)
    sents = split_sentences(content)
    
    # Choose question type based on Bloom level
    if bloom == "Knowledge":
        # Definition or description question
        if concepts:
            concept = concepts[variant % len(concepts)]
            question = f"Define and describe the key aspects of {concept}."
            model_answer = f"{concept} is defined as [definition]. Its key aspects include [aspect1], [aspect2], and [aspect3]."
        else:
            question = f"Describe the main features of {topic}."
            model_answer = f"The main features of {topic} include [feature1], [feature2], and [feature3]."
            
    elif bloom == "Comprehension":
        # Explanation or comparison question
        if relationships:
            rel = relationships[variant % len(relationships)]
            parts = rel.split(" - ")
            if len(parts) == 2:
                question = f"Explain the relationship between {parts[0]} and {parts[1]}."
                model_answer = f"The relationship between {parts[0]} and {parts[1]} can be explained as [explanation]. This relationship is important because [importance]."
            else:
                question = f"Explain the main principles of {topic}."
                model_answer = f"The main principles of {topic} include [principle1], [principle2], and [principle3]. These principles work together to [function]."
        else:
            question = f"Explain the main principles of {topic}."
            model_answer = f"The main principles of {topic} include [principle1], [principle2], and [principle3]. These principles work together to [function]."
            
    elif bloom == "Application":
        # Application or scenario-based question
        question = f"Describe a practical scenario where {topic} could be applied and explain the expected outcomes."
        model_answer = f"In a practical scenario, {topic} could be applied to [application]. The expected outcomes would include [outcome1], [outcome2], and [outcome3]."
        
    elif bloom == "Analysis":
        # Analysis or breakdown question
        question = f"Analyze the components of {topic} and explain how they interact with each other."
        model_answer = f"The components of {topic} include [component1], [component2], and [component3]. These components interact by [interaction]. This interaction leads to [result]."
        
    elif bloom == "Synthesis":
        # Synthesis or combination question
        question = f"Propose a new approach that combines {topic} with another concept to solve a specific problem."
        model_answer = f"A new approach could combine {topic} with [another concept] to solve [problem]. This approach would work by [mechanism] and would offer benefits such as [benefit1], [benefit2], and [benefit3]."
        
    else:  # Evaluation
        # Evaluation or judgment question
        question = f"Evaluate the effectiveness of {topic} and discuss its limitations and potential improvements."
        model_answer = f"The effectiveness of {topic} can be evaluated based on [criteria1], [criteria2], and [criteria3]. Its limitations include [limitation1] and [limitation2]. Potential improvements could include [improvement1] and [improvement2]."
    
    # Adjust complexity based on difficulty
    if difficulty == "easy":
        # For easy questions, add more guidance
        question += " Provide specific examples to support your answer."
    elif difficulty == "hard":
        # For hard questions, add more complexity
        question += " Consider multiple perspectives and potential counterarguments in your response."
    
    return {
        "question": question,
        "model_answer": model_answer,
        "difficulty": difficulty,
        "bloom_level": bloom,
        "topic": topic  # Keep topic for internal reference but not shown in question
    }

# ----------------------
# Local model caller (attempt Ollama), with graceful fallback
# ----------------------
def call_local_model_ollama(model: str, prompt: str, timeout: int = 120) -> str:
    cmd = [OLLAMA_CMD, "run", model]
    try:
        proc = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
        return proc.stdout.decode("utf-8", errors="ignore")

    except FileNotFoundError:
        raise FileNotFoundError("Ollama not found. Install from https://ollama.com/")
    except Exception:
        raise


def safe_parse_json_from_model(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {"status": "error", "raw_output": text[:1000]}
    

# ----------------------
# Ollama Generation
# ----------------------

def call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    try:
        result = subprocess.run(
            [OLLAMA_CMD, "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60
        )
        output = result.stdout.decode("utf-8").strip()
        return output
    except Exception as e:
        print(f"[WARN] Ollama call failed: {e}")
        return ""

def safe_json_extract(text: str) -> Dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return {}
    
    try:
        return json.loads(match.group(0))
    except:
        return {}


def generate_mcq_prompt(topic, content, difficulty, bloom, variant):
    return f"""
You are an expert academic question generator for university-level exams. Create a high-quality multiple-choice question based on the provided content.

CONTENT:
{content}

REQUIREMENTS:
1. Create a question appropriate for the {difficulty} difficulty level and {bloom} cognitive level.
2. The question should be clear, concise, and academically rigorous.
3. Provide exactly 4 options (A, B, C, D) with only one correct answer.
4. All options should be plausible but clearly differentiated.
5. The question should not reference the source material or indicate where it comes from.
6. The question should test understanding of the content, not just memorization.

BLOOM LEVEL GUIDELINES:
- Knowledge: Recall facts, terms, and basic concepts
- Comprehension: Explain ideas or concepts
- Application: Use information in new situations
- Analysis: Draw connections among ideas
- Synthesis: Create new ideas from multiple sources
- Evaluation: Justify a stand or decision

DIFFICULTY GUIDELINES:
- Easy: Direct application of concepts, minimal reasoning required
- Medium: Requires some analysis or synthesis of information
- Hard: Requires deep analysis, evaluation, or creative application

OUTPUT FORMAT:
{{
  "question": "Your question here",
  "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
  "answer": "A"  // The letter of the correct option
}}
"""


def generate_tf_prompt(topic, content, difficulty, bloom, variant):
    return f"""
You are an expert academic question generator for university-level exams. Create a high-quality true/false question based on the provided content.

CONTENT:
{content}

REQUIREMENTS:
1. Create a statement appropriate for the {difficulty} difficulty level and {bloom} cognitive level.
2. The statement should be clear, unambiguous, and academically rigorous.
3. The statement should be definitively true or false based on the content.
4. The statement should not reference the source material or indicate where it comes from.
5. The statement should test understanding of the content, not just memorization.

BLOOM LEVEL GUIDELINES:
- Knowledge: Recall facts, terms, and basic concepts
- Comprehension: Explain ideas or concepts
- Application: Use information in new situations
- Analysis: Draw connections among ideas
- Synthesis: Create new ideas from multiple sources
- Evaluation: Justify a stand or decision

DIFFICULTY GUIDELINES:
- Easy: Direct statement of fact
- Medium: Statement requiring some interpretation
- Hard: Statement requiring deep understanding or analysis

OUTPUT FORMAT:
{{
  "question": "Your statement here",
  "answer": true  // true or false
}}
"""

def generate_essay_prompt(topic, content, difficulty, bloom, variant):
    return f"""
You are an expert academic question generator for university-level exams. Create a high-quality essay question with a model answer based on the provided content.

CONTENT:
{content}

REQUIREMENTS:
1. Create a question appropriate for the {difficulty} difficulty level and {bloom} cognitive level.
2. The question should be clear, focused, and academically rigorous.
3. The question should require critical thinking and detailed response.
4. The question should not reference the source material or indicate where it comes from.
5. Provide a model answer that is comprehensive (5-10 sentences) and based on the content.

BLOOM LEVEL GUIDELINES:
- Knowledge: Describe or explain concepts
- Comprehension: Interpret or explain relationships
- Application: Apply concepts to new situations
- Analysis: Analyze components or relationships
- Synthesis: Create new ideas or perspectives
- Evaluation: Make judgments based on criteria

DIFFICULTY GUIDELINES:
- Easy: Straightforward explanation or description
- Medium: Analysis of relationships or moderate application
- Hard: Complex analysis, synthesis, or evaluation

OUTPUT FORMAT:
{{
  "question": "Your question here",
  "model_answer": "Your comprehensive model answer here (5-10 sentences)"
}}
"""


# ----------------------
# High-level generator
# ----------------------
def generate_questions_model(topic: str, content: str, difficulty: str, bloom: str,
                             use_ollama: bool, model_name: str, variants: int) -> Dict[str, List]:

    results = {"mcq": [], "tf": [], "essay": []}

    for v in range(variants):
        seed_variant = v + 5 # لتغيير كل مرة
        
        if use_ollama:
            # Generate MCQ
            mcq_prompt = generate_mcq_prompt(topic, content, difficulty, bloom, seed_variant)
            try:
                mcq_raw = call_local_model_ollama(model_name, mcq_prompt)
                mcq_parsed = safe_parse_json_from_model(mcq_raw)
                
                if "question" in mcq_parsed and "options" in mcq_parsed and "answer" in mcq_parsed:
                    results["mcq"].append({
                        "question": mcq_parsed["question"],
                        "options": mcq_parsed["options"][:4],
                        "answer": mcq_parsed["answer"],
                        "difficulty": difficulty,
                        "bloom_level": bloom,
                        "topic": topic
                    })
                else:
                    # Fallback to local generator
                    results["mcq"].append(make_mcq_local(topic, content, difficulty, bloom, seed_variant))
            except Exception:
                # Fallback to local generator
                results["mcq"].append(make_mcq_local(topic, content, difficulty, bloom, seed_variant))
            
            # Generate TF
            tf_prompt = generate_tf_prompt(topic, content, difficulty, bloom, seed_variant)
            try:
                tf_raw = call_local_model_ollama(model_name, tf_prompt)
                tf_parsed = safe_parse_json_from_model(tf_raw)
                
                if "question" in tf_parsed and "answer" in tf_parsed:
                    results["tf"].append({
                        "question": tf_parsed["question"],
                        "answer": bool(tf_parsed["answer"]),
                        "difficulty": difficulty,
                        "bloom_level": bloom,
                        "topic": topic
                    })
                else:
                    # Fallback to local generator
                    results["tf"].append(make_tf_local(topic, content, difficulty, bloom, seed_variant))
            except Exception:
                # Fallback to local generator
                results["tf"].append(make_tf_local(topic, content, difficulty, bloom, seed_variant))
            
            # Generate Essay
            essay_prompt = generate_essay_prompt(topic, content, difficulty, bloom, seed_variant)
            try:
                essay_raw = call_local_model_ollama(model_name, essay_prompt)
                essay_parsed = safe_parse_json_from_model(essay_raw)
                
                if "question" in essay_parsed and "model_answer" in essay_parsed:
                    results["essay"].append({
                        "question": essay_parsed["question"],
                        "model_answer": essay_parsed["model_answer"],
                        "difficulty": difficulty,
                        "bloom_level": bloom,
                        "topic": topic
                    })
                else:
                    # Fallback to local generator
                    results["essay"].append(make_essay_local(topic, content, difficulty, bloom, seed_variant))
            except Exception:
                # Fallback to local generator
                results["essay"].append(make_essay_local(topic, content, difficulty, bloom, seed_variant))
        else:
            # Use local generators
            results["mcq"].append(make_mcq_local(topic, content, difficulty, bloom, seed_variant))
            results["tf"].append(make_tf_local(topic, content, difficulty, bloom, seed_variant))
            results["essay"].append(make_essay_local(topic, content, difficulty, bloom, seed_variant))

    return results

# ----------------------
# Load topics from output JSON files
# ----------------------
def load_topics_from_output(output_dir: str) -> Tuple[List[str], Dict[str, str]]:
    topics = []
    contents = {}

    for file in Path(output_dir).glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for s in data.get("sections", []):
                title = s.get("section_title", "").strip()
                text = s.get("section_text", "").strip()

                if title:
                    if title not in contents:
                        topics.append(title)
                        contents[title] = text
                    else:
                        contents[title] += "\n" + text

        except Exception as e:
            print(f"Warning reading {file}: {e}")

    return topics, contents


# ----------------------
# Save a bank
# ----------------------
def save_bank(bank: Dict, filename: str):
    Path(BANKS_DIR).mkdir(exist_ok=True)
    out = Path(BANKS_DIR) / filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved → {out}")


# ----------------------
# MAIN
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-ollama", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variants", default=2, type=int)
    args = parser.parse_args()

    use_ollama = args.use_ollama
    model_name = args.model
    variants = args.variants

    print("Loading topics...")
    topics, contents = load_topics_from_output(OUTPUT_DIR)

    full_bank = {}

    for diff in DIFFICULTIES:
        print(f"\n### Difficulty: {diff} ###")

        bank = {bl: {"mcq": [], "tf": [], "essay": []} for bl in BLOOM_LEVELS}

        for topic in topics:
            content = contents[topic]

            for bl in BLOOM_LEVELS:
                bundle = generate_questions_model(
                    topic, content, diff, bl,
                    use_ollama=use_ollama,
                    model_name=model_name,
                    variants=variants
                )

                bank[bl]["mcq"] += bundle["mcq"]
                bank[bl]["tf"] += bundle["tf"]
                bank[bl]["essay"] += bundle["essay"]

        save_bank(bank, f"{diff}_bank.json")
        full_bank[diff] = bank

    with open(COMBINED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(full_bank, f, indent=2, ensure_ascii=False)

    print("\nAll banks saved successfully.")


if __name__ == "__main__":
    main()