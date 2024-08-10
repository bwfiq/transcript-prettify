import json
import os
import sys
import time
import tiktoken
from openai import OpenAI

def load_transcript_from_json(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)
    transcript_text = " ".join([chunk["text"] for chunk in data["chunks"]])
    return transcript_text

def chunk_text(transcript_text, max_tokens, encoding, overlap=50):
    tokens = encoding.encode(transcript_text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(encoding.decode(chunk))
        
        start += max_tokens - overlap  # Shift start to allow overlap

    return chunks

def generate_corrected_transcript(client, transcript_text, context=""):
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Your task is to correct formatting and spelling discrepancies in the transcribed text. Start each line of dialogue on a new line with a timestamp. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided."
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
            model="gpt-4o",
            temperature=0.2,
            messages=messages
        )
    except Exception as e:
        print(f"Error generating corrected transcript: {e}")
        sys.exit(1)

    return response.choices[0].message.content

def summarize_text(client, text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You are a master summarizer that can provide clear and concise summaries."
                },
                {
                    "role": "user",
                    "content": f"Summarize the following text: {text}"
                }
            ]
        )
    except Exception as e:
        print(f"Error summarizing text: {e}")
        sys.exit(1)

    return response.choices[0].message.content

def save_corrected_transcript(output_file_path, corrected_text):
    try:
        with open(output_file_path, 'a') as file:  # Open file in append mode
            file.write(corrected_text)
    except Exception as e:
        print(f"Error saving corrected transcript: {e}")
        sys.exit(1)

def main(json_file_path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("API key not found. Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key
    )

    print("Loading transcript from JSON...")
    start_time = time.time()
    transcript_text = load_transcript_from_json(json_file_path)
    load_time = time.time() - start_time
    print(f"Transcript loaded in {load_time:.2f} seconds.")

    encoding = tiktoken.encoding_for_model("gpt-4")
    max_tokens = 1000
    overlap = 200  # Adjust overlap as needed

    print("Chunking transcript text...")
    start_time = time.time()
    chunks = chunk_text(transcript_text, max_tokens, encoding, overlap)
    chunking_time = time.time() - start_time
    print(f"Text chunked into {len(chunks)} chunks in {chunking_time:.2f} seconds.")

    output_file_path = os.path.splitext(json_file_path)[0] + "_corrected.txt"
    
    # Clear the output file before appending new content
    open(output_file_path, 'w').close()

    context = ""
    total_chunks = len(chunks)

    print("Starting transcript correction...")
    total_start_time = time.time()

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{total_chunks}...")

        start_time = time.time()
        corrected_chunk = generate_corrected_transcript(client, chunk, context)
        correction_time = time.time() - start_time
        print(f"Chunk {i+1} corrected in {correction_time:.2f} seconds.")

        start_time = time.time()
        save_corrected_transcript(output_file_path, corrected_chunk + "\n")
        save_time = time.time() - start_time
        print(f"Chunk {i+1} saved in {save_time:.2f} seconds.")

        context = summarize_text(client, corrected_chunk)
        print(f"Summary of transcript so far: {context}")

    total_time = time.time() - total_start_time
    print(f"Transcript correction completed in {total_time:.2f} seconds.")
    print(f"Corrected transcript saved to {output_file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <json_file_path>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    main(json_file_path)
