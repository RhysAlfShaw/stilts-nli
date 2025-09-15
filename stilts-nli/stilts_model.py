from threading import Thread
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)

TEMPERATURE = 0.3

stilts_models = {
    "stilts-hf": "RAShaw/stilts_gemma_2b_finetunned_prototype",
    "stilts-guff-f16": "RAShaw/gemma-2b-it-stilts-prototype-GGUF",
    "stilts-guff-8bit": "RAShaw/gemma-2b-it-stilts-prototype-GGUF-Q8_0",
    "stilts-guff-4bit": "RAShaw/gemma-2b-it-stilts-prototype-GGUF-Q8_K_M",
}


class StiltsModel:
    """
    A class to encapsulate the finetunned model for stilts command generation.
    """

    def __init__(
        self,
        model_name: str = "RAShaw/gemma-2b-stilts-prototype",
        # model_name: str = "/scratch/Rhys/stilts_models/gemma-2b-finetuned/final_model",
        inference_library: str = "llama_cpp",
        num_proc: int = 5,
        device: str = "cpu",
        precision: str = "8bit",
        force_download: bool = False,
    ):
        """
        Initializes the Model class.

        Args:
            model_name (str): The path or Hugging Face repository ID of the model.
        """
        self.model_name = model_name
        self.force_download = force_download
        if device == "cpu":
            print(
                "Warning: Running on CPU may be slow. Consider using llama_cpp for faster CPU inference or running on a GPU."
            )
            self.device = device

        elif device == "cuda":
            if torch.cuda.is_available():
                print("Using GPU for inference.")
                self.device = device
            else:
                print("CUDA is not available, falling back to CPU.")
                self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                print(
                    "Warning: Running on CPU may be slow. Consider using llama_cpp for faster CPU inference or running on a GPU."
                )

        self.device = device
        self.num_proc = num_proc
        self.model = None
        self.tokenizer = None
        self.inference_library = inference_library

        self.precision = precision
        if self.precision not in ["float16", "8bit", "4bit"]:
            raise ValueError("Precision must be one of 'float16', '8bit', or '4bit'.")

        if self.inference_library == "transformers":
            self.load_model_transformers()
        elif self.inference_library == "llama_cpp":
            self.model_name = "RAShaw/gemma-2b-it-stilts-prototype-GGUF"
            self.load_model_llama_cpp()
        else:
            raise ValueError(
                f"Unsupported inference library: {self.inference_library}, please use 'transformers' or 'llama_cpp'."
            )

    def load_model_transformers(self):
        """
        Loads the model and tokenizer from the specified path.
        """
        print(f"Loading model '{self.model_name}' onto {self.device}...")
        try:
            if self.precision == "8bit":

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

            elif self.precision == "4bit":

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_type=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = None
            if self.force_download:
                print("Forcing re-download of the model...")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                attn_implementation="eager",
                quantization_config=quantization_config,
                force_download=self.force_download,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("Model loaded successfully.")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_model_llama_cpp(self):
        print(f"Loading model '{self.model_name}' onto {self.device}...")
        try:
            from llama_cpp import Llama

            self.model = Llama.from_pretrained(
                repo_id=self.model_name,
                filename="gemma-2b-it-stilts-prototype.gguf",
                n_threads=self.num_proc,
                n_threads_batch=self.num_proc,
                n_batch=32,
                dtype="float16",
                n_ctx=1024,  # Set context size to 1024
                verbose=False,
            )
            self.tokenizer = None  # Assuming tokenizer is not needed for llama_cpp
            print("Model loaded successfully.")

        except ImportError:
            print(
                "llama_cpp library is not installed. Please install it to use this model."
            )
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_stream_llama_cpp(self, prompt: str, max_new_tokens: int = 500):
        """
        Generates a response from the model as a stream using llama_cpp.

        Args:
            prompt (str): The input text to the model.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Yields:
            str: The next chunk of generated text.
        """

        messages = [
            {"role": "user", "content": prompt},
        ]
        response_generator = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            stop=["<end_of_turn>"],
            stream=True,  # Control whether to stream the response
        )

        for chunk in response_generator:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                text = chunk["choices"][0].get("delta", {}).get("content", "")
                if text:
                    yield text

    def generate_stream_transformers(self, prompt: str, max_new_tokens: int = 500):
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
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        prompt = (
            """<bos><start_of_turn>user\n"""
            + prompt
            + """<end_of_turn>\n<start_of_turn>assistant\n"""
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        eos_token_id = self.tokenizer("<end_of_turn>")["input_ids"][-1]

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.95,
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def generate_stream(self, prompt: str, max_new_tokens: int = 500):
        """
        Generates a response from the model as a stream.

        """
        if self.inference_library == "transformers":
            return self.generate_stream_transformers(prompt, max_new_tokens)
        elif self.inference_library == "llama_cpp":
            return self.generate_stream_llama_cpp(prompt, max_new_tokens)
        else:
            raise ValueError(
                f"Unsupported inference library: {self.inference_library}, please use 'transformers' or 'llama_cpp'."
            )
