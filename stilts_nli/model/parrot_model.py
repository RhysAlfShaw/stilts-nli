import time
from .basemodel import BaseModel


class ParrotModel(BaseModel):
    """
    A mock model class for testing and development purposes.

    This class inherits from BaseModel and mimics the behavior of a real language model
    without loading any large files or performing actual inference. It returns a
    pre-defined, streaming text response, making it ideal for unit tests or
    for developing applications when a GPU is not available.
    """

    def __init__(self, **kwargs):
        # Set defaults for the test model
        kwargs.setdefault("model_name", "parrot-model")
        # Override library to a fixed value to ensure predictability
        kwargs["inference_library"] = "transformers"
        super().__init__(**kwargs)

    def _load_model(self):
        """
        Overrides the base method to simulate model loading.

        This prevents any actual model downloading or initialization. It simply
        prints a message and sets mock objects for the model and tokenizer.
        """
        print(
            f"--- MOCK: Simulating model load for '{self.model_name}' on device '{self.device}'. ---"
        )
        self.model = "mock_model_object"
        self.tokenizer = "mock_tokenizer_object"
        time.sleep(0.5)  # loading delay
        print("--- MOCK: Mock model loaded successfully. ---")

    def load_model_transformers(self):
        pass

    def load_model_llama_cpp(self):
        pass

    def generate_stream_transformers(self, prompt: str, max_new_tokens: int = 500):
        """
        Simulates a streaming text generation response.

        Yields words from a pre-defined string to mimic the behavior of a
        real model generating tokens one by one.
        """
        print(f"\n--- MOCK: Generating stream for prompt: '{prompt}' ---")

        if "stilts" in prompt.lower() or "concatenate" in prompt.lower():
            response_text = '[stilts_command_generation(description="Concatenate table1.fits and table2.fits into a new file named combined.fits")]'
        elif "hello" in prompt.lower():
            response_text = "Hello! I am a test model. How can I assist you today?"
        else:
            response_text = (
                f"This is a simulated streaming response to your prompt: '{prompt}'"
            )

        words = response_text.split()
        for i, word in enumerate(words):
            yield " " + word if i > 0 else word
            time.sleep(0.05)  # generation delay

    def generate_stream_llama_cpp(self, prompt: str, max_new_tokens: int = 500):
        """
        Provides a mock implementation for llama_cpp, though the __init__
        method forces the use of the transformers path for this test model.
        """
        yield f"This is a mock response from the llama_cpp generator for the prompt: '{prompt}'"

    def _system_prompt(self):
        """A mock system prompt for the test model."""
        return "You are a parrot"
