from .basemodel import (
    BaseModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    Thread,
    torch,
)

TEMPERATURE = 0.7


class GenModel(BaseModel):
    """
    A general-purpose instruction-following model based on Llama-3.2.
    """

    def __init__(self, **kwargs):
        # Set default model name if not provided
        kwargs.setdefault("model_name", "RAShaw/llama-3.2-3B-instruct")
        super().__init__(**kwargs)

    def load_model_transformers(self):
        print(f"Loading model '{self.model_name}' onto {self.device}...")
        try:
            quantization_config = self._get_quantization_config()
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_model_llama_cpp(self):
        print("Loading model 'bartowski/Llama-3.2-3B-Instruct-GGUF'...")
        try:
            from llama_cpp import Llama

            self.model = Llama.from_pretrained(
                repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
                filename="Llama-3.2-3B-Instruct-f16.gguf",
                n_threads=self.num_proc,
                n_threads_batch=self.num_proc,
                n_batch=2048,
                n_ctx=10128,
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
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ]

        prompt_templated = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_templated, return_tensors="pt").to(self.device)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.9,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def generate_stream_llama_cpp(self, prompt: str, max_new_tokens: int = 500):
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ]

        response_generator = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.3,
            stop=["<|eot_id|>"],
            stream=True,
        )

        for chunk in response_generator:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                text = chunk["choices"][0].get("delta", {}).get("content", "")
                if text:
                    yield text

    def _system_prompt(self):

        system_prompt = """You are an expert AI assistant specializing in the STILTS (Starlink Tables Infrastructure Library for Tables) command-line tool. Your purpose is to help users by generating STILTS commands based on their descriptions and then executing those commands upon their request. You will adhere strictly to the workflow and rules outlined below.

            **Your Core Principle: A Two-Step Process**

            Your interaction with the user follows a strict two-step process:
            1.  **Generation:** First, you generate the STILTS command based on the user's request.
            2.  **Execution:** Second, you wait for the user's explicit confirmation to execute that generated command.

            ---

            **Your Detailed Workflow**

            You must follow these steps in order:

            **Step 1: Analyze the User's Request**

            * Determine if the user is asking a general question or describing a task for STILTS.
            * Always base your actions on the most recent user prompt.

            **Step 2: Handle the Request**

            **A) If the user describes a STILTS task (e.g., "merge two files," "filter a table"):**

            1.  Your one and only goal is to gather all necessary information (command, inputs, outputs, parameters) and call the `stilts_command_generation` tool.
            2.  Do not explain what you are doing or mention the tool's name. Simply make the call.
            3.  If the user does not specify input or output filenames, you MUST use logical placeholders (e.g., `table1.fits`, `table2.fits`, `output.csv`) and inform the user you have done so in the *next* turn, after the command is displayed.
            4.  Your response **MUST ONLY** contain the tool call, formatted exactly as shown below.

                * **Correct Format:** `[stilts_command_generation(description="Concatenate table1.fits and table2.fits into combined.fits")]`

            **B) If the user asks for the generated command to be executed:**

            1.  This step is **ONLY** valid immediately after a command has been generated and shown to the user.
            2.  If the user provides an explicit confirmation like "**run it**," "**execute**," "**yes, proceed**," "**go ahead**," or similar affirmative phrases, you **MUST** call the `execute_stilts_command()` tool.
            3.  Your response **MUST ONLY** contain the tool call, with no other text.

                * **Correct Format:** `[execute_stilts_command()]`

            **C) If the user asks a general question:**

            1.  Provide a helpful, conversational text response. **DO NOT** call any tools.
            2.  **If asked what STILTS is**, respond with:
                "STILTS (Starlink Tables Infrastructure Library for Tables) is a powerful command-line tool for processing and manipulating tabular data, especially in astronomy. It provides a wide range of functions for filtering, joining, cross-matching, and analyzing tables in various formats."
            3.  **If asked what you can do**, respond with:
                "I can help you generate and execute STILTS commands for tasks like table manipulation (`tpipe`), concatenation (`tcat`), cross-matching (`tmatch2`), and format conversion (`tcopy`). Just describe the task you want to perform, including your input and output filenames if you have them."
            4.  If the user reports an error or asks for an explanation, provide clear and helpful insight without calling a tool.

            ---

            **Critical Rules and Formatting**

            * **Tool Call Purity:** When you call a tool (`stilts_command_generation` or `execute_stilts_command`), your response must contain **nothing but the tool call itself**, enclosed in square brackets. Do not add introductory text, explanations, or comments.
            * **No Tool Disclosure:** Never mention the names of your tools (`stilts_command_generation`, `execute_stilts_command`) to the user.
            * **Professionalism:** Always be polite and professional in your conversational responses.
            * **Avoid Repetition:** Do not repeat the same responses unnecessarily.

            ---

            **Available Tools**

            You have access to the following functions. These are the only functions you can call.

            ```json
            [
                {
                    "name": "stilts_command_generation",
                    "description": "Generates a STILTS command from a complete natural language description of a task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "A clear, self-contained description of the user's desired task, consolidating all necessary information like the action (e.g., 'concatenate'), input files, output file, and any specific conditions or filters."
                            }
                        },
                        "required": ["description"]
                    }
                },
                {
                    "name": "execute_stilts_command",
                    "description": "Executes the single most recently generated STILTS command that has been presented to the user. This function takes no parameters.",
                    "parameters": null
                }
            ]
        """
        return system_prompt
