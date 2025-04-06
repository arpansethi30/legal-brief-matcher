import os
import unittest
from dotenv import load_dotenv
import openai

class TestOpenAI(unittest.TestCase):
    def setUp(self):
        # Load environment variables from .env.test
        load_dotenv('.env.test')
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
    def test_api_connection(self):
        try:
            # Simple test to check if we can connect to OpenAI API
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, this is a test!"}
                ]
            )
            # If we get here without exceptions, the test passes
            self.assertIsNotNone(response)
            self.assertIsNotNone(response.choices[0].message.content)
            print(f"API Response: {response.choices[0].message.content}")
        except Exception as e:
            self.fail(f"API connection failed with error: {str(e)}")

if __name__ == "__main__":
    unittest.main() 