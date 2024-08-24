import os
import time
import concurrent.futures
import random
import math
from openai_client import OpenAIClient
from utils import format_timestamp, load_transcript_from_json, chunk_text, post_process_transcript

class TranscriptProcessor:
    def __init__(self, api_key, model_name, transcript_context):
        self.client = OpenAIClient(api_key)
        self.encoding = self.client.get_encoding(model_name)
        self.model_name = model_name
        self.transcript_context = transcript_context
        self.cumulative_tokens = 0
        self.max_retries = 5
        self.base_delay = 0.4  # Base delay in seconds for rate limit retry

    def process_text_with_prompt(self, system_prompt, assistant_prompt, user_prompt, text, temperature=0.2):
        """Process text using OpenAI API with specified prompts."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": assistant_prompt},
            {"role": "user", "content": f"{user_prompt}: {text}"}
        ]
        return self.retry_request(lambda: self.client.generate_response(self.model_name, messages, temperature))

    def retry_request(self, request_func):
        """Retry request with exponential backoff."""
        retries = 0
        while retries < self.max_retries:
            try:
                response_text, tokens_used = request_func()
                self.cumulative_tokens += tokens_used
                return response_text, tokens_used
            except Exception as e:
                if 'Rate limit reached' in str(e):
                    wait_time = self.base_delay * (2 ** retries) + random.uniform(0, 0.1)
                    print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    raise e
        raise Exception("Max retries reached. Request failed.")

    def process_chunk(self, chunk, system_prompt, summary_prompt, context, index):
        """Process a single chunk and return the corrected text and updated context."""
        corrected_chunk, tokens_used = self.process_text_with_prompt(
            system_prompt, "", "Correct the following transcript: ", chunk
        )
        summary, _ = self.process_text_with_prompt(
            summary_prompt, "", "Summarize the following text: ", context + "\n" + corrected_chunk
        )
        return index, corrected_chunk, summary, tokens_used

    def process_transcript(self, json_file_path, chunk_size):
        """Main function to process, correct, and summarize the transcript."""
        transcript_text = load_transcript_from_json(json_file_path)
        chunks = chunk_text(transcript_text, chunk_size, self.encoding)
        num_chunks = len(chunks)
        output_file_path = os.path.splitext(json_file_path)[0] + "_corrected.txt"
        open(output_file_path, 'w').close()

        system_prompt = (
            f"You are a helpful assistant tasked with correcting formatting and spelling discrepancies in transcribed text. Your corrections should include only necessary punctuation, such as periods and commas, and appropriate capitalization. Ensure that each line maintains the format [HH:MM:SS - HH:MM:SS] text. Refer to the following context to accurately spell names, jargon, and other relevant terms that might appear in the transcript: {self.transcript_context}"
        )
        
        summary_prompt = (
            "You are a professional summarizer who can create a concise and comprehensive summary of any provided text, be it an article, post, conversation, or passage, while adhering to these guidelines: Craft a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity and conciseness. Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects. Rely strictly on the provided text, without including external information. Format the summary in paragraph form for easy understanding."
        )

        context = ""
        summary = ""

        try:
            for i, chunk in enumerate(chunks):
                start_time = time.time()
                chunk_tokens = 0
                corrected_chunk, tokens_used = self.process_text_with_prompt(
                    system_prompt, "", "Correct the following transcript: ", chunk
                )
                chunk_tokens += tokens_used
                self.save_corrected_transcript(output_file_path, corrected_chunk)

                context = summary + "\n" + corrected_chunk
                summary, tokens_used = self.process_text_with_prompt(
                    summary_prompt, "", "Summarize the following text: ", context
                )
                chunk_tokens += tokens_used
                
                print(f"Processed chunk {i+1}/{len(chunks)} using {chunk_tokens} tokens in {round(time.time() - start_time, 2)}s.")
            print(f"Transcript correction completed.")
        except KeyboardInterrupt:
            print(f"Transcript correction interrupted.")

        print(f"Cumulative tokens used: {self.cumulative_tokens}")
        print(f"Corrected transcript saved to {output_file_path}")
        print(f"Summary: {summary}")

    @staticmethod
    def save_corrected_transcript(output_file_path, corrected_text):
        """Append corrected transcript to the output file."""
        with open(output_file_path, 'a') as file:
            file.write(post_process_transcript(corrected_text))
