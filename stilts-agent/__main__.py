## main cli loop for stilts-agent.
import subprocess
import re
import logging
import json

from stilts_model import StiltsModel
from gen_model import GenModel 



# Add this at the beginning of your script
logging.getLogger("transformers").setLevel(logging.ERROR)

colors = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "reset": "\033[0m",
    "bold": "\033[1m",
    "underline": "\033[4m"
}

class CLI:
    def __init__(self):
        self.stilts_model = StiltsModel()
        self.gen_model = GenModel()
        print(f"""
        {colors['green']}{colors['bold']}
        Welcome to the Stilts Natural Language Interface!
        {colors['reset']}
        This tool allows you to generate STILTS commands and execute them using a natural language.
        You can ask the model to create commands based on your prompts.{colors['bold']}
        Type 'help/h' for guidence, 'clear/c' to clear the message history, 'quit/q' to exit.{colors['reset']}
        Save message history to a file type 'save/s'.
        """)
        self.message_history = []

    def add_to_message_history(self, message):
        """
        Adds a message to the message history.
        """
        self.message_history.append(message)

    def cli_loop(self):
        while True:
            self.get_input()
            if self.input.lower() == 'exit' or self.input.lower() == 'quit' or self.input.lower() == 'q':
                print(f"{colors['red']}Exiting CLI.{colors['reset']}")
                break

            elif self.input.lower() == 'help' or self.input.lower() == 'h':
                self._help()
                continue

            elif self.input.lower() == 'clear' or self.input.lower() == 'c':
                self.message_history = []
                print(f"{colors['red']}Message history cleared.{colors['reset']}")
                continue

            elif self.input.lower() == 'save' or self.input.lower() == 's':
                filename = input("Enter filename to save message history (without extension): ")
                # save history as JSON 
                with open(f"{filename}.json", "w") as f:
                    json.dump(self.message_history, f, indent=4)
                print(f"{colors['green']}Message history saved to {filename}.json{colors['reset']}")
                continue

            self.add_to_message_history({"role": "user", "content": self.input})

            command = self.gen_model.generate_stream(self.message_history)
            full_chunks = ""
            is_tool_call = False
            print("\n")
            # Stream the response from the general model.
            # If it's a regular text response, print it as it comes.
            # If it's a tool call (detected by '['), stop printing and just accumulate it.
            for chunk in command:
                full_chunks += chunk
                # Once we detect a tool call, we stop printing for the rest of the stream.
                if not is_tool_call and '[' in chunk:
                    is_tool_call = True

                if not is_tool_call:
                    print(chunk, end='', flush=True)
            
            if not is_tool_call:   
                print("\n")
            # print("TESING: ", full_chunks.strip())
            self.add_to_message_history({"role": "assistant", "content": full_chunks})
            
            gen_model_responce = full_chunks.strip()

            # check if there is more than one tool call in the response



            if "stilts_command_generation" in gen_model_responce:
                # Use regex for robust parsing of the tool call
                matches = re.findall(r"stilts_command_generation\s*\(\s*description\s*=\s*['\"](.*?)['\"]\s*\)", gen_model_responce, re.DOTALL)
                if matches:
                    for description in matches:
                        # print(f"Generating STILTS command for description: {description}")
                        stilts_command = self.stilts_model.generate_stream(description)
                        full_command = ""
                        for chunk in stilts_command:
                            print(chunk, end='', flush=True)
                            full_command += chunk

                        print("\n")

                        command_explanation = self.stilts_model.generate_stream(
                            f"Explain the following STILTS command: {full_command}"
                        )
                        full_explanation = ""
                        for chunk in command_explanation:
                            print(chunk, end='', flush=True)
                            full_explanation += chunk
                        print("\n")
                        self.add_to_message_history({
                            "role": "assistant",
                            "content": full_command + "\n\n" + full_explanation
                        })
                else:
                    print(f"{colors['red']}Error: Could not parse description from LLM tool call response.{colors['reset']}")
                    continue
            
            if "execute_stilts_command" in gen_model_responce:
                # Use regex for robust parsing of the tool call
                matches = re.findall(r"execute_stilts_command\s*\(\s*stilts_command\s*=\s*['\"](.*?)['\"]\s*\)", gen_model_responce, re.DOTALL)
                if matches:
                    for command in matches:
                        # print(f"Executing STILTS command: {command}")
                        returned_output = self.eval_execute_command(command)
                        self.add_to_message_history({
                            "role": "python",
                            "content": f"{returned_output}"
                        })
                else:
                    print(f"{colors['red']}Error: Could not parse command from LLM tool call response.{colors['reset']}")
                    continue

            


                
    def is_responce_function_call(self, response):
        """Check if the response is a function call."""
        return "stilts_command_generation" in response

    def greating(self):
        print("Welcome to the Stilts CLI!")

    def get_input(self):
        """ask the user for a command"""
        self.input = input(">> ")
    
    def run(self):
        # start loop for the CLI
        self.cli_loop()

    def _help(self):
        print("Example prompts:")
        print("1. ############")
        print("Prompt: 'Create a command to match catalogues input.fits and input2.fits using RA and dec columns to within 1 arcsec'.")
        print("2. ############")
        print("Prompt: 'How can I convert from a fits file to a csv file?'")

    def eval_execute_command(self, command):
        """execute the command"""
        should_execute = input(f"\n\nDo you want to execute this? (y/n): ")
        if should_execute.lower() in ['yes', 'y']:
            returned_out = self.execute_command(command)
        else:
            print("Command execution skipped.")
        
        return returned_out

    
    def execute_command(self, command):
        """execute the command using subprocess"""
        try:
            print(f"{colors['yellow']}")
            run = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            print(f"{colors['reset']}")
            print(f"{colors['green']}Command executed successfully.{colors['reset']}")
            return run.stdout

        except subprocess.CalledProcessError as e:
            print(e.stderr)
            print(f"{colors['red']}Error executing command: {e}{colors['reset']}")
            return e.stderr
        

def main():
    cli = CLI()
    cli.greating()
    cli.run()

if __name__ == "__main__":
    main()