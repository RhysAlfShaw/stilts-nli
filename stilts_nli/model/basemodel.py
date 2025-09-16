# base_model.py
import torch
from threading import Thread
from abc import ABC, abstractmethod

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)


class BaseModel(ABC):
    """
    A base class for language models to handle common setup and dispatching logic.

    This class manages device selection (CPU/CUDA), precision validation,
    and the overall workflow for loading models and generating text via different
    inference libraries. Full implementations must be provided by subclasses.
    """

    def __init__(
        self,
        model_name: str,
        inference_library: str = "transformers",
        num_proc: int = 5,
        device: str = "cpu",
        precision: str = "8bit",
    ):
        self.model_name = model_name
        self.inference_library = inference_library
        self.num_proc = num_proc
        self.precision = precision

        self.model = None
        self.tokenizer = None

        self._setup_device(device)
        self._validate_precision()
        self._load_model()

    def _setup_device(self, requested_device: str):
        """Sets up the computation device (CPU or CUDA)."""
        if requested_device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            print("Using GPU for inference.")
        elif requested_device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
            print("Using Apple Silicon GPU (MPS) for inference.")
        else:
            if requested_device == "cuda":
                print("CUDA is not available, falling back to CPU.")
            self.device = "cpu"
            print(
                "Warning: Running on CPU may be slow. Consider using llama_cpp for "
                "faster CPU inference or running on a GPU."
            )
            torch.set_num_threads(self.num_proc)

    def _validate_precision(self):
        """Validates that the specified precision is supported."""
        if self.precision not in ["float16", "8bit", "4bit"]:
            raise ValueError("Precision must be one of 'float16', '8bit', or '4bit'.")

    def _load_model(self):
        """Dispatches to the correct model loading method based on the inference library."""
        if self.inference_library == "transformers":
            self.load_model_transformers()
        elif self.inference_library == "llama_cpp":
            self.load_model_llama_cpp()
        else:
            raise ValueError(
                f"Unsupported inference library: {self.inference_library}. "
                "Please use 'transformers' or 'llama_cpp'."
            )

    @abstractmethod
    def load_model_transformers(self):
        """Abstract method to load a model using the Transformers library."""
        raise NotImplementedError("Subclasses must implement 'load_model_transformers'")

    @abstractmethod
    def load_model_llama_cpp(self):
        """Abstract method to load a model using the llama-cpp-python library."""
        raise NotImplementedError("Subclasses must implement 'load_model_llama_cpp'")

    def generate_stream(self, prompt: str, max_new_tokens: int = 500):
        """
        Generates a response from the model as a stream.

        This method acts as a dispatcher to the library-specific generation method.
        """
        if self.inference_library == "transformers":
            return self.generate_stream_transformers(prompt, max_new_tokens)
        elif self.inference_library == "llama_cpp":
            return self.generate_stream_llama_cpp(prompt, max_new_tokens)
        else:
            raise ValueError(
                f"Unsupported inference library: {self.inference_library}. "
                "Please use 'transformers' or 'llama_cpp'."
            )

    @abstractmethod
    def generate_stream_transformers(self, prompt: str, max_new_tokens: int = 500):
        """Abstract method for streaming generation with Transformers."""
        raise NotImplementedError(
            "Subclasses must implement 'generate_stream_transformers'"
        )

    @abstractmethod
    def generate_stream_llama_cpp(self, prompt: str, max_new_tokens: int = 500):
        """Abstract method for streaming generation with llama-cpp-python."""
        raise NotImplementedError(
            "Subclasses must implement 'generate_stream_llama_cpp'"
        )

    def _get_quantization_config(self) -> BitsAndBytesConfig | None:
        """Helper to create BitsAndBytesConfig based on precision."""
        if self.precision == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        elif self.precision == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_type=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        return None
