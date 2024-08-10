import json
import os
import sys
import time
import tiktoken
from openai import OpenAI
from datetime import timedelta

def format_timestamp(seconds):
    if seconds is None:
        return "00:00:00"  # Default to 00:00:00 if the timestamp is None
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def load_transcript_from_json(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)

    # Concatenate the text with formatted timestamps on each new line
    transcript_lines = [
        f"[{format_timestamp(chunk['timestamp'][0])} - {format_timestamp(chunk['timestamp'][1])}] {chunk['text']}"
        for chunk in data.get("chunks", [])
    ]
    
    transcript_text = "\n".join(transcript_lines)
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

def count_tokens(encoding, messages):
    total_tokens = 0
    for message in messages:
        total_tokens += len(encoding.encode(message['content']))
    return total_tokens

def generate_corrected_transcript(client, model_name, transcript_text, context="", encoding=None):
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Your task is to correct formatting and spelling discrepancies in the transcribed text. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided. Each line should have the format [HH:MM:SS - HH:MM:SS] text."
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
        
        # Count tokens before making the API call
        if encoding:
            total_tokens = count_tokens(encoding, messages)
            # print(f"Total tokens (context + generated) for this call: {total_tokens}")
        
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            messages=messages
        )
    except Exception as e:
        print(f"Error generating corrected transcript: {e}")
        sys.exit(1)

    return response.choices[0].message.content, total_tokens

def summarize_text(client, model_name, text, encoding=None):
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a master summarizer that can provide clear and concise summaries."
            },
            {
                "role": "user",
                "content": f"Summarize the following text: {text}"
            }
        ]
        
        # Count tokens before making the API call
        if encoding:
            total_tokens = count_tokens(encoding, messages)
            # print(f"Total tokens (context + generated) for this call: {total_tokens}")
        
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            messages=messages
        )
    except Exception as e:
        print(f"Error summarizing text: {e}")
        sys.exit(1)

    return response.choices[0].message.content, total_tokens

def save_corrected_transcript(output_file_path, corrected_text):
    try:
        with open(output_file_path, 'a') as file:  # Open file in append mode
            file.write(corrected_text)
    except Exception as e:
        print(f"Error saving corrected transcript: {e}")
        sys.exit(1)

def main(json_file_path, model_name):
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
    overlap = 0  # Adjust overlap as needed

    print("Chunking transcript text...")
    start_time = time.time()
    chunks = chunk_text(transcript_text, max_tokens, encoding, overlap)
    chunking_time = time.time() - start_time
    print(f"Text chunked into {len(chunks)} chunks in {chunking_time:.2f} seconds.")

    output_file_path = os.path.splitext(json_file_path)[0] + "_corrected.txt"
    
    # Clear the output file before appending new content
    open(output_file_path, 'w').close()

    context = ""
    summary = ""
    total_chunks = len(chunks)
    cumulative_tokens = 0  # Initialize cumulative token counter

    print("Starting transcript correction...")
    total_start_time = time.time()

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{total_chunks}...")

        start_time = time.time()
        corrected_chunk, tokens_used = generate_corrected_transcript(client, model_name, chunk, context, encoding)
        cumulative_tokens += tokens_used  # Update cumulative token counter
        correction_time = time.time() - start_time
        print(f"Chunk {i+1} corrected in {correction_time:.2f} seconds.")

        start_time = time.time()
        save_corrected_transcript(output_file_path, corrected_chunk + "\n")
        save_time = time.time() - start_time
        print(f"Chunk {i+1} saved in {save_time:.2f} seconds.")

        # Update context with the summary and the current corrected chunk
        context = summary + "\n" + corrected_chunk
        summary, tokens_used = summarize_text(client, model_name, context, encoding)
        cumulative_tokens += tokens_used  # Update cumulative token counter
        print(f"Summary of transcript so far: {summary}")
        print(f"Cumulative tokens used so far: {cumulative_tokens}")

    total_time = time.time() - total_start_time
    print(f"Transcript correction completed in {total_time:.2f} seconds.")
    print(f"Corrected transcript saved to {output_file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <json_file_path> <model_name>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    model_name = sys.argv[2]
    main(json_file_path, model_name)
