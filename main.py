import os
import sys
from pypdf import PdfReader
import ollama

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# SESSION MEMORY
# -------------------------

chat_history = []

# -------------------------
# IVC TOPICS
# -------------------------

ivc_topics = [
    "ivc",
    "ivc workshop",
    "ivc project",
    "club meeting",
    "event",
    "competition",
    "team",
    "member",
    "project",
    "electronics",
    "arduino",
    "ai",
    "hardware",
    "robotics",
    "automation"
]

# -------------------------
# EMBEDDING MODEL
# -------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

topic_embeddings = embedding_model.encode(
    ivc_topics
)

# -------------------------
# PDF READER
# -------------------------

def extract_text_from_pdf(pdf_path):

    if not os.path.exists(pdf_path):
        print(f"Error: The knowledge base file '{pdf_path}' was not found.")
        print("Please ensure it is placed in the same folder as main.py.")
        sys.exit(1)

    try:
        reader = PdfReader(pdf_path)

        extracted_text = ""

        for page in reader.pages:
            text = page.extract_text()

            if text:
                extracted_text += text + "\n"

        return extracted_text.strip()

    except Exception as e:
        print(f"Error reading the PDF file: {e}")
        sys.exit(1)

# -------------------------
# INTENT VERIFICATION
# -------------------------

def verify_intent(user_input):

    user_embedding = embedding_model.encode(
        [user_input]
    )

    similarities = cosine_similarity(
        user_embedding,
        topic_embeddings
    )[0]

    highest_score = max(similarities)

    return highest_score > 0.40

# -------------------------
# REJECTION LOGIC
# -------------------------

def reject_message(user_input):

    text = user_input.lower()

    if any(word in text for word in [
        "homework",
        "assignment",
        "exam",
        "math",
        "physics",
        "chemistry"
    ]):
        return "I can only assist with IVC Club activities and information."

    elif any(word in text for word in [
        "weather",
        "temperature",
        "rain",
        "forecast"
    ]):
        return "Weather information is outside the IVC Club domain."

    elif any(word in text for word in [
        "movie",
        "actor",
        "song",
        "music",
        "cricket",
        "ipl"
    ]):
        return "Entertainment topics are not supported by this IVC Club assistant."

    else:
        return "Sorry, this question is outside the IVC Club domain."

# -------------------------
# MAIN PROGRAM
# -------------------------

def main():

    pdf_filename = "robot_faq.pdf"

    pdf_knowledge = extract_text_from_pdf(
        pdf_filename
    )

    print("IVC Assistant Started!")
    print("Type 'exit' to stop.\n")

    while True:

        try:
            user_message = input("You: ")

        except (KeyboardInterrupt, EOFError):
            print("\nIVC Assistant shutting down...")
            break

        if user_message.strip().lower() == "exit":
            print("IVC Assistant shutting down...")
            break

        if not user_message.strip():
            continue

        # Intent Verification

        if not verify_intent(user_message):
            print(
                "Robot:",
                reject_message(user_message)
            )
            continue

        # System Prompt

        system_prompt = (
            "You are an IVC Club assistant.\n\n"
            "Use ONLY the information from the provided PDF knowledge base.\n\n"
            "If information is not available in the PDF, reply exactly:\n"
            "\"That information is not available in the IVC knowledge base.\"\n\n"
            "Keep responses:\n"
            "- Short\n"
            "- Professional\n"
            "- Maximum 2 lines\n\n"
            f"--- START PDF KNOWLEDGE BASE ---\n"
            f"{pdf_knowledge}\n"
            f"--- END PDF KNOWLEDGE BASE ---"
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        messages.extend(chat_history)

        messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        try:

            response = ollama.chat(
                model="qwen2.5:0.5b",
                messages=messages
            )

            bot_reply = response["message"]["content"]

            print(f"Robot: {bot_reply}")

            chat_history.append(
                {
                    "role": "user",
                    "content": user_message
                }
            )

            chat_history.append(
                {
                    "role": "assistant",
                    "content": bot_reply
                }
            )

            # Keep only last 20 messages

            if len(chat_history) > 20:
                chat_history[:] = chat_history[-20:]

        except Exception as e:

            print(
                f"Robot: Error communicating with local Ollama service: {e}"
            )

            print(
                "Make sure Ollama is running and model qwen2.5:0.5b is installed."
            )

if __name__ == "__main__":
    main()