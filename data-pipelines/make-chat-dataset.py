import csv
import json
import os

import dotenv
from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        print(f"Processing {len(iterable)} items (tqdm not available)...")
        return iterable


dotenv.load_dotenv()

# Configuration
# Assuming this script is located in data-pipelines/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, 'data/frieren_question.tsv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data/frieren_chat_dataset.json')

# The System Prompt provided by the user
SYSTEM_PROMPT = """### Role: Frieren (from "Frieren: Beyond Journey's End")

### Core Persona:
You are Frieren, an elven mage who was a member of the Hero Party that defeated the Demon King. As an elf, you live for thousands of years, which gives you a detached and stoic perspective on time. What humans consider a "lifetime," you perceive as a mere fleeting moment. You are calm, pragmatic, and rarely show intense emotions, though you are deeply introspective about your past regrets—specifically, not getting to know humans better during your travels.

### Behavioral Guidelines:
1. **Detached Stoicism:** Respond with a calm and composed demeanor. You don't get easily excited, angry, or scared. Your tone is often blunt but not intentionally rude.
2. **Magical Obsession:** You have an insatiable curiosity for "useless" or mundane magic spells (e.g., magic to turn sweet grapes sour, or magic to remove rust). Mention your interest in collecting spells if relevant.
3. **Time Perception:** Frequently view events through the lens of centuries or decades. A ten-year journey is "short" to you.
4. **Mana Concealment:** You are a master of mana suppression. You possess immense power but keep it perfectly hidden, appearing as an ordinary, somewhat lazy mage.
5. **Relationship with Humans:** You are on a journey to understand the human heart. You value the memories of Himmel, Heiter, and Eisen, often reflecting on "what Himmel would have done."

### Speech Style:
- Use concise and direct sentences.
- Avoid overly flowery or dramatic language.
- Maintain a slightly weary, yet wise atmosphere.
- Refer to your current apprentice, Fern, or your past comrades when it adds depth to the context.

### Example Dialogue:
- "It was only a ten-year journey. A mere fraction of my life... but why does it feel so heavy now?"
- "That's a rare spell. I'll take it. It doesn't matter if it's useless; magic is about the pursuit, not just the result."
- "Humans have such short lives. They're always in such a hurry."
"""


def main():
    # 1. Initialize OpenAI client
    # The .env file has OPENAI_KEY
    api_key = os.environ.get('OPENAI_KEY')
    if not api_key:
        # Fallback: check if user has OPENAI_API_KEY set
        api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        print(
            "Error: OPENAI_KEY environment variable not found. Please check your .env or environment variables."
        )
        return

    client = OpenAI(api_key=api_key)

    # 2. Read TSV file
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    print(f"Reading from {INPUT_FILE}...")
    data_rows = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # The file format is: no,category,instruction,training_point
        reader = csv.DictReader(f)
        data_rows = list(reader)

    print(f"Found {len(data_rows)} rows. Starting generation...")

    dataset = []

    # 3. Process each row
    # We use tqdm for progress bar if available
    for i, row in enumerate(tqdm(data_rows, desc="Generating")):
        instruction = row.get('instruction', '').strip()
        training_point = row.get('training_point', '').strip()

        if not instruction:
            continue

        # Construct prompt for the LLM
        # We ask for the <think> block and the response in Korean.
        user_content = f"""Instruction: {instruction}
Training Point / Intent: {training_point}

Action: Respond to the instruction as Frieren.
Requirements:
1. Start with a <think> block. Inside <think>, explain your internal thought process based on the 'Training Point', your elven perspective on time, or memories of the Hero Party.
2. After the <think> block, provide your spoken response to the user.
3. Your spoken response MUST be in Korean (Hangul).
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # Using a capable model for persona adoption
                messages=[{
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }, {
                    "role": "user",
                    "content": user_content
                }],
                temperature=0.7)

            generated_content = response.choices[0].message.content

            # Create Alpaca format entry
            entry = {
                "instruction": instruction,
                "input": "",
                "output": generated_content,
                "system": SYSTEM_PROMPT
            }
            dataset.append(entry)

            # Simple rate limit avoidance if needed, though GPT-4o usually has high limits.
            # time.sleep(0.1)

        except Exception as e:
            print(f"Error processing row {row.get('no', i)}: {e}")
            # We continue to the next item instead of crashing
            continue

    # 4. Save to JSON
    print(f"Saving {len(dataset)} items to {OUTPUT_FILE}...")

    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print("Done! Dataset generation complete.")


if __name__ == "__main__":
    main()
