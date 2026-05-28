import ollama

# Allowed topics
allowed_topics = [
    "robotics",
    "club",
    "event",
    "meeting",
    "workshop",
    "project"
]

# Chat memory
chat_history = []

print("Robot Assistant Started!")
print("Type 'exit' to stop.\n")

while True:

    # Take user input
    user_input = input("You: ").lower()

    # Exit condition
    if user_input == "exit":
        print("Robot shutting down...")
        break

    # Intent checking
    allowed = False

    for topic in allowed_topics:
        if topic in user_input:
            allowed = True

    # If allowed
    if allowed:

        # Save memory
        chat_history.append(f"User: {user_input}")

        # Combine conversation
        full_prompt = "\n".join(chat_history)

        # Ask Ollama AI
        response = ollama.chat(
            model='qwen2.5:0.5b',
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a robotics club assistant. Only answer robotics club related questions.'
                },
                {
                    'role': 'user',
                    'content': full_prompt
                }
            ]
        )

        # Extract AI reply
        bot_reply = response['message']['content']

        print("Robot:", bot_reply)

        # Save bot response
        chat_history.append(f"Bot: {bot_reply}")

    else:

        print("Robot: Sorry, I only answer club-related questions.")

