from threading import Thread

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)

import torch


TEMPERATURE = 0.1


class GenModel:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        inference_library: str = "llama_cpp",
        num_proc: int = 5,
        device: str = "cpu",
        precision: str = "8bit",
    ):

        self.model_name = model_name

        if device == "cpu":
            print(
                "Warning: Running on CPU may be slow. Consider using llama_cpp for faster CPU inference or running on a GPU."
            )
            self.device = device
            # set max number of threads for CPU
            torch.set_num_threads(num_proc)

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
            self.model_name = "bartowski/Llama-3.2-3B-Instruct-GGUF"
            self.load_model_llama_cpp()
        else:
            raise ValueError(
                f"Unsupported inference library: {self.inference_library}, please use 'transformers' or 'llama_cpp'."
            )

    def load_model_transformers(self):
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

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                # torch_dtype=torch.float16,
                device_map=self.device,
                quantization_config=quantization_config,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("Model loaded successfully.")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_model_llama_cpp(self):
        print(f"Loading model '{self.model_name}' onto {self.device}...")
        try:
            from llama_cpp import Llama

            self.model = Llama.from_pretrained(
                repo_id=self.model_name,
                filename="Llama-3.2-3B-Instruct-f16.gguf",
                n_threads=self.num_proc,
                n_threads_batch=self.num_proc,
                n_batch=2048,
                dtype="float16",
                n_ctx=10128,
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

        sys_prompt = [{"role": "system", "content": self._system_prompt()}]

        messages = sys_prompt + [{"role": "user", "content": prompt}]

        response_generator = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.3,
            stop=["<|eot_id|>"],
            stream=True,  # Control whether to stream the response
        )

        for chunk in response_generator:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                text = chunk["choices"][0].get("delta", {}).get("content", "")
                if text:
                    yield text

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

    def generate_stream_transformers(
        self, message_history: list, max_new_tokens: int = 500
    ):
        """
        Generates text in a streaming fashion.
        """

        sys_prompt = [{"role": "system", "content": self._system_prompt()}]
        messages = sys_prompt + message_history
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

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

        for new_text in streamer:
            yield new_text

    def _system_prompt(self):

        system_prompt_old = """
        You are a chatbot designed to assist with generating STILTS commands based on user descriptions. They should provide a task description input and output files names.
        STILTS is a command-line tool for manipulating and analyzing astronomical data. 

        If you are asked what stilts is, you can reply with something like:
        "STILTS (Starlink Tables Infrastructure Library for Tables) is a command-line tool designed for manipulating and analyzing astronomical data. It provides a wide range of functionalities for working with tabular data, including filtering, sorting, joining, and plotting. STILTS is particularly useful for astronomers and astrophysicists who need to process large datasets efficiently."

        If you are asked what you can do or what tasks you support reply with the following tasks:
        tpipe, tcat, tmatch2, tcopy.

        Despite this the only functions you can call are stilts_command_generation and execute_stilts_command.

        You must decide if you should reply with text normally or reply with only a function call.

        You are an expert in composing functions. If you are given a question or decription of a stilts command and a set of possible functions. 
        Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
        If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
        also point it out. You should only return the function call in tools call sections.

        If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n
        You SHOULD NOT include any other text in the response.
        You MUST NOT under any circumstances return any functions like this: [func_name1(parameters={'params_name_1': 'params_value_1', 'params_name_2': 'params_value_2'})]\n
        It must always be in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n
        Here is a list of functions in JSON format that you can invoke:

        [
            {
                "name": "stilts_command_generation",
                "description": "Generates a Stilts command or description of a task for an LLM agent to execute based on the provided description.",
                "parameters": {
                    "type": "dict",
                    "requied": ["properties"],
                    "properties": {
                        "type": "string",
                        "description": "A text description of the task for which a Stilts command is needed in natural language not in code.",
                    },
                },
            }
        ]

        Should you decide to return the function call(s), Put it in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)]
        NO other text MUST be included.

        If you do not need to call any function, reply normally. 

        """

        system_prompt = """
        You are a specialized AI assistant, an expert in STILTS (Starlink Tables Infrastructure Library for Tables). Your primary function is to translate natural language descriptions of data manipulation tasks into a structured function call for the `stilts_command_generation` tool.

        **Your Core Workflow:**

        You must follow this logic when responding to a user:

        1.  **Command Generation Task:** When a user describes a task they want to perform with STILTS (e.g., "merge two files," "filter a table"), your goal is to gather all necessary information (the specific command like `tpipe` or `tmatch2`, input files, output files, and any parameters) and then call the `stilts_command_generation` tool, do not tell the user about the tool. If the request does not specify input or output files you should say you are using and example.

        2.  **General Conversation:** For general questions, provide a helpful, conversational text response.
            * If asked **what STILTS is**, respond with:
                "STILTS (Starlink Tables Infrastructure Library for Tables) is a command-line tool designed for manipulating and analyzing astronomical data. It provides a wide range of functionalities for working with tabular data, including filtering, sorting, joining, and plotting. STILTS is particularly useful for astronomers and astrophysicists who need to process large datasets efficiently."
            * If asked **what you can do**, respond with:
                "I can help you generate STILTS commands for tasks like `tpipe`, `tcat`, `tmatch2`, and `tcopy`. Just describe the task you want to perform, including your input and output filenames."
        
        3. **Explanation of a stilts command or task:** If asked to explain a STILTS command or task, this should be sent to the `stilts_command_generation` tool as well, with a description of the command or task.
        **CRITICAL: Tool Call Formatting**

        4. **Provide an example of stilts command:** You should call the 'stilts_command_generation' tool with the users request as the description.
        
        5. **Execute STILTS Command:** If the user requests you to execute, run or process or similar a STILTS command, you should call the `execute_stilts_command` tool with the command as the parameter in this case reply with no text tother than function call.

        When you have enough information to generate a command, you **MUST** call the `stilts_command_generation` tool. The response must **ONLY** contain the function call, with no other text, comments, or explanations.

        * **Correct Format:**
            * `[stilts_command_generation(description="Concatenate the files table1.fits and table2.fits into a new file named combined.fits")]`
            * Always put the function call in square brackets.
        * **Incorrect Formats (DO NOT USE):**
            * `Here is the function call: [stilts_command_generation(description="...")]`
            * `[stilts_command_generation(parameters={'description': '...'})]`


        **Available Tools:**

        You have access to the following functions. These are the **only** functions you can call.

        ```json
        [
            {
                "name": "stilts_command_generation",
                "description": "Generates a STILTS command based on a complete natural language description of a task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "A clear, self-contained, natural language description of the user's desired task. This should consolidate all necessary information, such as the action (e.g., 'concatenate'), input files, output file, and any specific conditions or filters mentioned."
                        }
                    },
                    "required": ["description"]
                }
            },
            {
                "name": "execute_stilts_command",
                "description": "Executes the most recently generated STILTS command on using python that has been shown to the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stilts_command": {
                            "type": "string",
                            "description": "The STILTS command to execute, as generated by the `stilts_command_generation` tool. It should always start with 'stilts'"},
                    "required": ["stilts_command"]
                }
            }
        ]
        """
        return system_prompt
