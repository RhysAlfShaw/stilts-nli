from .basemodel import (
    BaseModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    Thread,
    torch,
)

TEMPERATURE = 0.1


class StiltsModel(BaseModel):
    """
    A fine-tuned model for generating STILTS commands.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("model_name", "RAShaw/gemma-2b-stilts-prototype")
        super().__init__(**kwargs)

    def load_model_transformers(self):
        print(f"Loading model '{self.model_name}' onto {self.device}...")
        try:
            quantization_config = self._get_quantization_config()
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                attn_implementation="eager",
                quantization_config=quantization_config,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_model_llama_cpp(self):
        print("Loading model 'RAShaw/gemma-2b-it-stilts-prototype-GGUF'...")
        try:
            from llama_cpp import Llama

            self.model = Llama.from_pretrained(
                repo_id="RAShaw/gemma-2b-it-stilts-prototype-GGUF",
                filename="gemma-2b-it-stilts-prototype.gguf",
                n_threads=self.num_proc,
                n_threads_batch=self.num_proc,
                n_batch=32,
                n_ctx=1024,
                verbose=False,
            )
            print("Model loaded successfully.")
        except ImportError:
            print(
                "llama_cpp library is not installed. Please install it to use this model."
            )
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_stream_transformers(self, prompt: str, max_new_tokens: int = 500):
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        prompt_templated = (
            "<bos><start_of_turn>user\n"
            + prompt
            + "<end_of_turn>\n<start_of_turn>assistant\n"
        )

        inputs = self.tokenizer(prompt_templated, return_tensors="pt").to(self.device)
        eos_token_id = self.tokenizer("<end_of_turn>")["input_ids"][-1]

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def generate_stream_llama_cpp(self, prompt: str, max_new_tokens: int = 500):
        messages = [{"role": "user", "content": prompt}]
        response_generator = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.3,
            stop=["<end_of_turn>"],
            stream=True,
        )

        for chunk in response_generator:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                text = chunk["choices"][0].get("delta", {}).get("content", "")
                if text:
                    yield text
