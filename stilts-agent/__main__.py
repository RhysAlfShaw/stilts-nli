## main cli loop for stilts-agent.
import subprocess

from stilts_model import StiltsModel
from gen_model import GenModel 

import logging

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
    "reset": "\033[0m"
}

class CLI:
    def __init__(self):
        self.stilts_model = StiltsModel()
        self.gen_model = GenModel()
        print("""
        Welcome to the Stilts CLI!
        This CLI allows you to generate STILTS commands using a language model.
        You can ask the model to create commands based on your prompts.
        Type 'help' for examples or 'q' to exit.
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
                print("Exiting CLI.")
                break

            elif self.input.lower() == 'help':
                self._help()
                continue
                
            self.add_to_message_history({"role": "user", "content": self.input})

            command = self.gen_model.generate_stream(self.message_history)
            full_chunks = ""
            print("\n")
            for chunk in command:
                print(chunk, end=
                      '', flush=True)
                full_chunks += chunk
            print("\n")
            self.add_to_message_history({"role": "assistant", "content": full_chunks})
            
            gen_model_responce = full_chunks.strip()
            
            if "stilts_command_generation" in gen_model_responce:
                try:
                    description = gen_model_responce.split("properties='")[1].split("')")[0].strip("'\"")
                except IndexError:
                    try:
                        description = (
                            gen_model_responce.split("description=")[1].split(")")[0].strip("'\"")
                        )
                    except IndexError:
                        try:
                            description = (
                                gen_model_responce.split("type='")[1].split("')")[0].strip("'\"")
                            )
                        except IndexError:
                            print("Error: processing llm tool call response")

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
            self.execute_command(command)
        else:
            print("Command execution skipped.")

    
    def execute_command(self, command):
        """execute the command using subprocess"""
        subprocess.run(command, shell=True, check=True)
        print(
            "Finished running STILTS command, check TESTING_CATALOG for the resulting output."
        )
       

def main():
    cli = CLI()
    cli.greating()
    cli.run()

if __name__ == "__main__":
    main()