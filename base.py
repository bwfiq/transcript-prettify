import json
import os
import sys
import logging
import time
import tiktoken
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_transcript_from_json(json_file_path):
    """Load transcript text from a JSON file."""
    start_time = time.time()
    logging.info(f"Loading transcript from JSON file: {json_file_path}")
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        logging.error(f"Failed to load JSON file: {e}")
        sys.exit(1)
    
    # Concatenate all text from the transcript
    transcript_text = " ".join([chunk["text"] for chunk in data["chunks"]])
    elapsed_time = time.time() - start_time
    logging.info(f"Transcript loaded and concatenated in {elapsed_time:.2f} seconds.")
    return transcript_text

def chunk_text(transcript_text, max_tokens, encoding):
    """Chunk the transcript text to ensure it does not exceed max tokens."""
    tokens = encoding.encode(transcript_text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [encoding.decode(chunk) for chunk in chunks]

def generate_corrected_transcript(client, temperature, system_prompt, transcript_text, context=""):
    """Generate corrected transcript using OpenAI API."""
    start_time = time.time()
    logging.info("Generating corrected transcript using OpenAI API...")
    try:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        if context:
            messages.append({
                "role": "assistant",
                "content": context
            })
        messages.append({
            "role": "user",
            "content": transcript_text
        })
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Use the appropriate model
            temperature=temperature,
            messages=messages
        )
        elapsed_time = time.time() - start_time
        logging.info(f"Corrected transcript generated successfully in {elapsed_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Failed to generate corrected transcript: {e}")
        sys.exit(1)

    return response.choices[0].message.content

def summarize_text(client, temperature, system_prompt, text):
    """Summarize text using OpenAI API."""
    start_time = time.time()
    logging.info("Summarizing text using OpenAI API...")
    try:
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
                    "content": f"Summarize the following text: {text}"
                }
            ]
        )
        elapsed_time = time.time() - start_time
        logging.info(f"Summary generated successfully in {elapsed_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Failed to generate summary: {e}")
        sys.exit(1)

    return response.choices[0].message.content

def save_corrected_transcript(output_file_path, corrected_text):
    """Save the corrected transcript to a file."""
    logging.info(f"Saving corrected transcript to file: {output_file_path}")
    try:
        with open(output_file_path, 'w') as file:
            file.write(corrected_text)
        logging.info(f"Corrected transcript saved successfully at {output_file_path}.")
    except Exception as e:
        logging.error(f"Failed to save corrected transcript: {e}")
        sys.exit(1)

def main(json_file_path):
    start_time = time.time()

    # Ensure the OpenAI API key is set in the environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    # Initialize the OpenAI client
    client = OpenAI(
        api_key=api_key  # Explicitly passing the API key, even though it's the default
    )

    logging.info("OpenAI client initialized.")

    # Define the system prompt
    system_prompt = (
        "You are a helpful assistant that helps transcribe DnD sessions. Your task is to correct spelling discrepancies in the transcribed text. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided. Include timestamps and newlines. The player characters are Senzorin, Arche, Saziver, and Zel'eon. The system is DnD 5e. The setting is a contemporary fantasy world, and the session is set in a city called Ilrinia. The following words must be spelled correctly: Ilrinia, Senzorin, Arche, Saziver, Zel'eon, and Arcana."
    )

    # Load the transcript from the provided JSON file
    transcript_text = load_transcript_from_json(json_file_path)

    # Initialize the encoding for tokenization
    encoding = tiktoken.encoding_for_model("gpt-4o")

    # Define the max tokens per chunk (adjusted for some buffer to avoid cutting off mid-sentence)
    max_tokens = 10000

    # Chunk the transcript text
    chunks = chunk_text(transcript_text, max_tokens, encoding)
    
    corrected_text = ""
    context = ""
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {i + 1}/{len(chunks)}...")
        
        # Generate the corrected text for this chunk
        corrected_chunk = generate_corrected_transcript(client, 0, system_prompt, chunk, context)
        
        # Append to the overall corrected text
        corrected_text += corrected_chunk + "\n"
        
        # Summarize the corrected text so far
        context = summarize_text(client, 0, system_prompt, corrected_text)

    # Determine the output file path
    output_file_path = os.path.splitext(json_file_path)[0] + "_corrected.txt"

    # Save the corrected transcript to a file
    save_corrected_transcript(output_file_path, corrected_text)

    # Calculate total execution time
    total_time = time.time() - start_time
    logging.info(f"Total execution time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    # Ensure the script is called with a JSON file argument
    if len(sys.argv) != 2:
        logging.error("Usage: python correct_transcript.py <path_to_json_file>")
        sys.exit(1)
    
    # Get the JSON file path from the command line argument
    json_file_path = sys.argv[1]
    
    logging.info("Script started.")
    logging.info(f"Processing file: {json_file_path}")
    
    # Run the main function
    main(json_file_path)
    
    logging.info("Script finished.")
