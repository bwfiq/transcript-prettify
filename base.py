import json
import os
import sys
import time
import argparse
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

def generate_corrected_transcript(client, model_name, system_prompt, transcript_text, context="", encoding=None):
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
        
        # Count tokens before making the API call
        if encoding:
            total_tokens = count_tokens(encoding, messages)
            #print(f"Total tokens (context + generated) for this call: {total_tokens}")
        
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
                "content": "You are a professional summarizer who can create a concise and comprehensive summary of any provided text, be it an article, post, conversation, or passage, while adhering to these guidelines: Craft a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity and conciseness. Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects. Rely strictly on the provided text, without including external information. Format the summary in paragraph form for easy understanding."
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
            # Process the text to add new lines for Markdown display
            processed_text = post_process_transcript(corrected_text)
            file.write(processed_text)
    except Exception as e:
        print(f"Error saving corrected transcript: {e}")
        sys.exit(1)

def post_process_transcript(text):
    """Adds a new line after each line of dialogue for Markdown display."""
    lines = text.splitlines()
    processed_lines = [line + "\n" for line in lines]  # Add a blank line after each line
    processed_lines.append("")
    return "\n".join(processed_lines)

def main(json_file_path, model_name, transcript_context, chunk_size):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("API key not found. Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    start_time = time.time()
    transcript_text = load_transcript_from_json(json_file_path)
    load_time = time.time() - start_time
    print(f"Transcript loaded in {load_time:.2f} seconds.")

    encoding = tiktoken.encoding_for_model(model_name)
    max_tokens = chunk_size
    overlap = 0  # Adjust overlap as needed

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

    # Construct the system prompt using the provided words list and transcript description
    system_prompt = (
        "You are a helpful assistant tasked with correcting formatting and spelling discrepancies in the provided transcribed text. "
        "Your corrections should include only necessary punctuation, such as periods and commas, and appropriate capitalization. "
        "Ensure that each line maintains the format [HH:MM:SS - HH:MM:SS] text. "
        "Refer to the following context to accurately spell names, jargon, and other relevant terms that might appear in the transcript: "
        f"{transcript_context}"
    )

    total_start_time = time.time()
    transcriptCompleted = False
    
    try:
        for i, chunk in enumerate(chunks):
            start_time = time.time()

            corrected_chunk, tokens_used = generate_corrected_transcript(client, model_name, system_prompt, chunk, context, encoding)
            cumulative_tokens += tokens_used
            
            save_corrected_transcript(output_file_path, corrected_chunk + "\n")
            
            context = summary + "\n" + corrected_chunk
            summary, tokens_used = summarize_text(client, model_name, context, encoding)
            
            cumulative_tokens += tokens_used

            print(f"Processed chunk {i+1}/{total_chunks} in {time.time() - total_start_time:.2f} seconds.")
        transcriptCompleted = True
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        transcriptCompleted = False

    total_time = time.time() - total_start_time
    if (transcriptCompleted):
        print(f"Transcript correction completed in {total_time:.2f} seconds.")
    else:
        print(f"Transcript correction interrupted after {total_time:.2f} seconds.")
    print(f"Cumulative tokens used so far: {cumulative_tokens}")
    print(f"Corrected transcript saved to {output_file_path}")
    print(f"Summary: {summary}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and correct transcript using GPT-4 or GPT-3.5-Turbo.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the transcript JSON file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Name of the OpenAI model to use (e.g., 'gpt-4'). Default is gpt-4o-mini.")
    parser.add_argument("--info_path", type=str, default="", help="Path to the text file containing the transcript description and other relevant information. Default is an empty string.")
    parser.add_argument("--chunk_size", type=int, default=2000, help="Size (in tokens) of the chunks to split the text into for passing to the API. Default is 2000 tokens.")

    args = parser.parse_args()

    # Read the content of the information file if provided
    if args.info_path:
        with open(args.info_path, 'r') as file:
            transcript_context = file.read().strip()
    else:
        transcript_context = ""

    main(
        json_file_path=args.json_path,
        model_name=args.model_name,
        transcript_context=transcript_context,
        chunk_size=args.chunk_size
    )

