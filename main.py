import os
import sys
import re
from pypdf import PdfReader
import ollama

# Initialize conversation history memory as a global list as required.
# This will store the chat messages to maintain context across turns.
chat_history = []

def extract_text_from_pdf(pdf_path):
    """
    Extracts and compiles all text from a given PDF knowledge base.
    Uses pypdf.PdfReader to read pages sequentially.
    """
    # Check if the PDF file exists in the directory before attempting to open it
    if not os.path.exists(pdf_path):
        print(f"Error: The knowledge base file '{pdf_path}' was not found.")
        print("Please ensure it is placed in the same folder as main.py.")
        sys.exit(1)
        
    try:
        reader = PdfReader(pdf_path)
        extracted_text = ""
        # Loop through all pages and concatenate the text
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        return extracted_text.strip()
    except Exception as e:
        print(f"Error reading the PDF file: {e}")
        sys.exit(1)

def verify_intent(user_input):
    """
    Checks if the user prompt is related to allowed topics.
    Allowed topics: robotics, robot, workshop, club, project, AI, event, sensors, hardware, meeting.
    Uses regex to check for 'ai' as a whole word to avoid false positives (e.g. 'rain', 'pain').
    """
    cleaned_input = user_input.lower()
    
    # 1. Clean and split user input into individual words for strict matches (like 'ai')
    words = re.findall(r'\b\w+\b', cleaned_input)
    
    # 2. Define the list of allowed topics (case-insensitive)
    allowed_topics = [
        "robotics",
        "robot",
        "workshop",
        "club",
        "project",
        "ai",
        "event",
        "sensors",
        "hardware",
        "meeting"
    ]
    
    # 3. Check matches
    for topic in allowed_topics:
        if topic == "ai":
            # For 'ai', we perform a whole-word match to prevent false matches
            if "ai" in words:
                return True
        else:
            # For other topics, substring matches are allowed (e.g., 'robot' matches 'robots')
            if topic in cleaned_input:
                return True
                
    return False

def main():
    # Path to the PDF knowledge base file
    pdf_filename = "robot_faq.pdf"
    
    # Extract the context from the PDF file
    pdf_knowledge = extract_text_from_pdf(pdf_filename)
    
    # Display startup messages
    print("Robot Assistant Started!")
    print("Type 'exit' to stop.\n")
    
    # Main continuous chatbot loop
    while True:
        try:
            # Get question from the user
            user_message = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nRobot shutting down...")
            break
            
        # Check if the user wants to exit the chat
        if user_message.strip().lower() == "exit":
            print("Robot shutting down...")
            break
            
        # Skip empty inputs
        if not user_message.strip():
            continue
            
        # Intent verification step
        if not verify_intent(user_message):
            print("Robot: sorry , This is not my cup of tea")
            continue
            
        # Define the system prompt incorporating the PDF knowledge base
        system_prompt = (
            "You are a robotics club assistant.\n\n"
            "Use ONLY the information from the provided PDF knowledge base.\n"
            "If information is not available in the PDF, say:\n"
            "\"That information is not available in the robotics knowledge base.\"\n\n"
            "Keep responses:\n"
            "* short\n"
            "* professional\n"
            "* maximum 2 lines\n\n"
            f"--- START PDF KNOWLEDGE BASE ---\n{pdf_knowledge}\n--- END PDF KNOWLEDGE BASE ---"
        )
        
        # Build the message history list for Ollama
        # We start with the system prompt to enforce the rules on every query
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation memory from the history
        messages.extend(chat_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        try:
            # Query the local Ollama AI model (qwen2.5:0.5b) offline
            response = ollama.chat(
                model="qwen2.5:0.5b",
                messages=messages
            )
            
            # Extract the response message content
            bot_reply = response["message"]["content"]
            
            # Print the AI's reply
            print(f"Robot: {bot_reply}")
            
            # Save the exchange in the chat history memory list
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": bot_reply})
            
        except Exception as e:
            print(f"Robot: Error communicating with local Ollama service: {e}")
            print("Make sure Ollama is running and model qwen2.5:0.5b is loaded.")

if __name__ == "__main__":
    main()
