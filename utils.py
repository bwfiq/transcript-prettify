import json
from datetime import timedelta

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    if seconds is None:
        return "00:00:00"
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def load_transcript_from_json(json_file_path):
    """Load and format transcript from JSON file."""
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    transcript_lines = [
        f"[{format_timestamp(chunk['timestamp'][0])} - {format_timestamp(chunk['timestamp'][1])}] {chunk['text']}"
        for chunk in data.get("chunks", [])
    ]
    return "\n".join(transcript_lines)

def chunk_text(text, max_tokens, encoding, overlap=0):
    """Split text into chunks with token limits and overlap."""
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunks.append(encoding.decode(tokens[start:end]))
        start += max_tokens - overlap
    return chunks

def post_process_transcript(text):
    """Add new lines for Markdown formatting."""
    return "\n".join(line + "\n" for line in text.splitlines()) + "\n"
