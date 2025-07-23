from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch

device = "cuda"
class Model:
    def __init__(self):
        self.model_name = "/scratch/Rhys/stilts_models/gemma-2b-it-finetuned/final_model" # we set this for it to be grabbed from huggingface or other model repositories.
        self.load_model()

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map=device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        

    def generate_command(self, prompt):
        # IDEALLY we would have a stream of the output here and running with llama_cpp, but for now we will just return the full output.
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        response = (
            response.split("<end_of_turn>")[1].strip().split("<end_of_turn>")[0].strip()
        )
        response = response.replace("<start_of_turn>model", "").strip()

        return response
