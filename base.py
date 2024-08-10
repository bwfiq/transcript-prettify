import json
import os
import sys
from openai import OpenAI

def load_transcript_from_json(json_file_path):
    """Load transcript text from a JSON file."""
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    # Concatenate all text from the transcript
    transcript_text = " ".join([chunk["text"] for chunk in data["chunks"]])
    return transcript_text

def generate_corrected_transcript(client, temperature, system_prompt, transcript_text):
    """Generate corrected transcript using OpenAI API."""
    response = client.chat.completions.create(
        model="gpt-4o",  # Use the appropriate model
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": transcript_text
            }
        ]
    )
    return response.choices[0].message.content

def main(json_file_path):
    # Ensure the OpenAI API key is set in the environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    # Initialize the OpenAI client
    client = OpenAI(
        api_key=api_key  # Explicitly passing the API key, even though it's the default
    )

    # Define the system prompt
    system_prompt = (
        "You are a helpful assistant that help Dungeons and Dragons DMs transcribe their session recordings. Your task is to correct spelling discrepancies in the transcribed text. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided."
    )

    # Load the transcript from the provided JSON file
    transcript_text = load_transcript_from_json(json_file_path)

    # Generate and print the corrected text
    corrected_text = generate_corrected_transcript(client, 0, system_prompt, transcript_text)
    print(corrected_text)

if __name__ == "__main__":
    # Ensure the script is called with a JSON file argument
    if len(sys.argv) != 2:
        print("Usage: python correct_transcript.py <path_to_json_file>")
        sys.exit(1)
    
    # Get the JSON file path from the command line argument
    json_file_path = sys.argv[1]
    
    # Run the main function
    main(json_file_path)
