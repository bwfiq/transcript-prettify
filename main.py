import os
import sys
import argparse
from transcript_processor import TranscriptProcessor

def main():
    parser = argparse.ArgumentParser(description="Process and correct transcript using OpenAI.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the transcript JSON file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="OpenAI model name (e.g., 'gpt-4').")
    parser.add_argument("--info_path", type=str, default="", help="Path to the context file (optional).")
    parser.add_argument("--chunk_size", type=int, default=2000, help="Size of chunks in tokens.")

    args = parser.parse_args()
    transcript_context = ""
    if args.info_path:
        with open(args.info_path, 'r') as file:
            transcript_context = file.read().strip()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("API key not found. Set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    processor = TranscriptProcessor(api_key, args.model_name, transcript_context)
    processor.process_transcript(args.json_path, args.chunk_size)

if __name__ == "__main__":
    main()
