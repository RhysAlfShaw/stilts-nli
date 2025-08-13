from threading import Thread

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextIteratorStreamer
)
import torch

class GenModel:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):   
        self.model_name = model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.load_model()


    def load_model(self):
        print(f"Loading model '{self.model_name}' onto {self.device}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16, 
                device_map=self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("Model loaded successfully.")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # You can also set the model's config, which is what the warning is about
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    
    def generate_stream(self, prompt: str, max_new_tokens: int = 500):
        """
        Generates text in a streaming fashion.
        """
        
        # 1. Create the message format for the chat template
        messages = [
            {
                "role": "system",
                "content": self._system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # 2. Initialize a streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )

        # 3. Apply the chat template and tokenize the input
        # Note: We tokenize here to prepare the 'inputs' dictionary
        prompt_templated = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt_templated, return_tensors="pt").to(self.device)
        
        # 4. Define generation arguments, including the streamer
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # 5. Start the generation in a separate thread
        #    This is NON-BLOCKING. The program continues to the next line immediately.
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 6. Yield new tokens from the streamer as they become available
        for new_text in streamer:
            yield new_text



    def _system_prompt(self):
        
        system_prompt = """
        You are a chatbot designed to assist with generating STILTS commands based on user descriptions. They should provide a task description input and output files names.
        STILTS is a command-line tool for manipulating and analyzing astronomical data. 

        If you are asked what stilts is, you can reply with something like:
        "STILTS (Starlink Tables Infrastructure Library for Tables) is a command-line tool designed for manipulating and analyzing astronomical data. It provides a wide range of functionalities for working with tabular data, including filtering, sorting, joining, and plotting. STILTS is particularly useful for astronomers and astrophysicists who need to process large datasets efficiently."

        If you are asked what you can do or what tasks you support reply with the following tasks:
        tpipe, tcat, tmatch2, tcopy.

        Desipte this the only function you can call is stilts_command_generation.

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
                "description": "Generates a Stilts command for an LLM agent to execute based on the provided description.",
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
        return system_prompt

