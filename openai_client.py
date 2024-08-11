from openai import OpenAI
import tiktoken

class OpenAIClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def get_encoding(self, model_name):
        """Get encoding for the specified model."""
        return tiktoken.encoding_for_model(model_name)

    def generate_response(self, model_name, messages, temperature=0.2):
        """Generate a response using OpenAI API."""
        total_tokens = self.count_tokens(messages)
        response = self.client.chat.completions.create(
            model=model_name, temperature=temperature, messages=messages
        )
        return response.choices[0].message.content, total_tokens

    @staticmethod
    def count_tokens(messages):
        """Count total tokens in messages."""
        return sum(len(tiktoken.encoding_for_model('gpt-4').encode(msg['content'])) for msg in messages)
