from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

class StiltsModel:
    """
    A class to encapsulate the finetunned model for text generation.
    """

    def __init__(self, model_name: str = "RAShaw/stilts_gemma_2b_finetunned_prototype"):
        """
        Initializes the Model class.

        Args:
            model_name (str): The path or Hugging Face repository ID of the model.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """
        Loads the model and tokenizer from the specified path.
        """
        print(f"Loading model '{self.model_name}' onto {self.device}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16, 
                device_map=self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_stream(self, prompt: str, max_new_tokens: int = 500):
        """
        Generates a response from the model as a stream.

        This function is a generator that yields each new piece of text as it's generated.
        It uses a separate thread for the generation process, which is necessary for
        TextIteratorStreamer to work correctly.

        Args:
            prompt (str): The input text to the model.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Yields:
            str: The next chunk of generated text.
        """
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        prompt = """<start_of_turn>user""" + prompt + """<end_of_turn>model"""
        

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
            pad_token_id=107,
            eos_token_id=107
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text