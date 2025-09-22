import setproctitle

setproctitle.setproctitle("stilts-nli")

## main cli loop for stilts-agent.
import subprocess
import re
import logging
import json
import argparse
import os

from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter

from stilts_nli.model.stilts_model import StiltsModel
from stilts_nli.model.gen_model import GenModel
from stilts_nli.model.parrot_model import ParrotModel
from stilts_nli.utils.completion import CombinedCompleter

logging.getLogger("transformers").setLevel(logging.ERROR)


prompt_session_history = PromptSession()
options_completer = CombinedCompleter()

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
    "underline": "\033[4m",
    "italic": "\033[3m",
}


class CLI:
    def __init__(
        self,
        inference_library,
        num_proc,
        device: str,
        stilts_model_only: bool = False,
        precision_stilts_model: str = "float16",
        precision_gen_model: str = "8bit",
        force_download: bool = False,
        test_mode: bool = False,
    ):
        self.precision_stilts_model = precision_stilts_model
        self.precision_gen_model = precision_gen_model
        self.stilts_model_only = stilts_model_only
        self.inference_library = inference_library
        self.num_proc = num_proc
        self.force_download = force_download

        print(
            f"Using inference library: {self.inference_library}, number of processes: {self.num_proc}, device: {device}"
        )

        if device not in ["cpu", "cuda", "mps"]:
            raise ValueError("Device must be 'cpu' or 'cuda'.")

        self.device = device

        if test_mode:
            self.stilts_model = ParrotModel(
                inference_library=self.inference_library,
                num_proc=self.num_proc,
                device=self.device,
                precision=self.precision_stilts_model,
            )
        else:
            self.stilts_model = StiltsModel(
                inference_library=self.inference_library,
                num_proc=self.num_proc,
                device=self.device,
                precision=self.precision_stilts_model,
            )
        if self.stilts_model_only:
            print(
                f"""
            {colors['green']}{colors['bold']}
            Welcome to the Stilts Natural Language Interface! (Stilts Model Only Mode)
            {colors['reset']}
            This tool allows you to generate STILTS commands and execute them using a natural language.
            You can ask the model to create commands based on your prompts.
            {colors['bold']}Type 'help/h' for guidence, 'quit/q' to exit.{colors['bold']}

            Need more help? Visit: {colors['blue']}{colors['underline']}{colors["bold"]}https://www.star.bristol.ac.uk/~mbt/stilts/{colors['reset']}

            {colors['italic']}LLM generated commands should be checked before execution.{colors['reset']}   

            """
            )

        else:
            if test_mode:
                self.gen_model = ParrotModel(
                    inference_library=self.inference_library,
                    num_proc=self.num_proc,
                    device=self.device,
                    precision=self.precision_gen_model,
                )
            else:
                self.gen_model = GenModel(
                    inference_library=self.inference_library,
                    num_proc=self.num_proc,
                    device=self.device,
                    precision=self.precision_gen_model,
                )
            print(
                f"""
            {colors['green']}{colors['bold']}
            Welcome to the Stilts Natural Language Interface!
            {colors['reset']}
            This tool allows you to generate STILTS commands and execute them using a natural language.
            You can ask the model to create commands based on your prompts.
            Once it generates a command ask it to execute it.{colors['bold']}
            Type 'help/h' for guidence, 'clear/c' to clear the message history, 'quit/q' to exit.{colors['reset']}
            Save message history to a file type 'save/s'.
            Need more help? Visit: {colors['blue']}{colors['underline']}{colors["bold"]}https://www.star.bristol.ac.uk/~mbt/stilts/{colors['reset']} 
            
            {colors['italic']}LLM generated commands should be checked before execution.{colors['reset']}
            
            """
            )
            self.message_history = []

    def stilts_model_loop(self):
        options_completer_stilts = CombinedCompleter(reduced_option=True)
        while True:
            description = prompt_session_history.prompt(
                ">> ",
                auto_suggest=AutoSuggestFromHistory(),
                completer=options_completer_stilts,
            )
            if description.lower() in ["exit", "quit", "q"]:
                print(f"{colors['red']}Exiting Stilts Model Loop.{colors['reset']}")
                break

            stilts_command = self.stilts_model.generate_stream(description)
            full_command = ""
            print("\nResponse:\n")
            for chunk in stilts_command:
                print(chunk, end="", flush=True)
                full_command += chunk
            print("\n")

    def add_to_message_history(self, message):
        """
        Adds a message to the message history.
        """
        self.message_history.append(message)

    def cli_loop(self):
        while True:
            self.get_input()
            if (
                self.input.lower() == "exit"
                or self.input.lower() == "quit"
                or self.input.lower() == "q"
            ):
                print(f"{colors['red']}Exiting CLI.{colors['reset']}")
                break

            elif self.input.lower() == "help" or self.input.lower() == "h":
                self._help()
                continue

            elif self.input.lower() == "clear" or self.input.lower() == "c":
                self.message_history = []
                print(f"{colors['red']}Message history cleared.{colors['reset']}")
                continue

            elif self.input.lower() == "save" or self.input.lower() == "s":
                filename = prompt_session_history.prompt(
                    "Enter filename to save message history (without extension): ",
                    auto_suggest=AutoSuggestFromHistory(),
                    completer=None,
                )

                with open(f"{filename}.json", "w") as f:
                    json.dump(self.message_history, f, indent=4)
                print(
                    f"{colors['green']}Message history saved to {filename}.json{colors['reset']}"
                )
                continue

            self.add_to_message_history({"role": "user", "content": self.input})

            command = self.gen_model.generate_stream(self.message_history)
            full_chunks = ""
            is_tool_call = False
            print("\n")

            for chunk in command:
                full_chunks += chunk
                if not is_tool_call and "[" in chunk:
                    is_tool_call = True

                if not is_tool_call:
                    print(chunk, end="", flush=True)

            if not is_tool_call:
                print("\n")
            self.add_to_message_history({"role": "assistant", "content": full_chunks})

            gen_model_responce = full_chunks.strip()

            if "stilts_command_generation" in gen_model_responce:
                matches = re.findall(
                    r"stilts_command_generation\s*\(\s*description\s*=\s*['\"](.*?)['\"]\s*\)",
                    gen_model_responce,
                    re.DOTALL,
                )
                if matches:
                    for description in matches:
                        stilts_command = self.stilts_model.generate_stream(description)
                        full_command = ""
                        for chunk in stilts_command:
                            print(chunk, end="", flush=True)
                            full_command += chunk

                        print("\n")

                        command_explanation = self.stilts_model.generate_stream(
                            f"Explain the following STILTS command: {full_command}"
                        )
                        full_explanation = ""
                        for chunk in command_explanation:
                            print(chunk, end="", flush=True)
                            full_explanation += chunk
                        print("\n")
                        self.add_to_message_history(
                            {
                                "role": "assistant",
                                "content": full_command + "\n\n" + full_explanation,
                            }
                        )
                else:
                    print(
                        f"{colors['red']}Error: Could not parse description from LLM tool call response.{colors['reset']}"
                    )
                    continue

            if "execute_stilts_command" in gen_model_responce:
                matches = re.findall(
                    r"execute_stilts_command\s*\(\s*stilts_command\s*=\s*['\"](.*?)['\"]\s*\)",
                    gen_model_responce,
                    re.DOTALL,
                )
                if matches:
                    for command in matches:
                        returned_output = self.eval_execute_command(command)
                        self.add_to_message_history(
                            {"role": "python", "content": f"{returned_output}"}
                        )
                else:
                    print(
                        f"{colors['red']}Error: Could not parse command from LLM tool call response.{colors['reset']}"
                    )
                    continue

    def is_responce_function_call(self, response):
        """Check if the response is a function call."""
        return "stilts_command_generation" in response

    def greating(self):
        print("Welcome to the Stilts CLI!")

    def get_input(self):
        """ask the user for a command"""
        self.input = prompt_session_history.prompt(
            ">> ",
            auto_suggest=AutoSuggestFromHistory(),
            completer=options_completer,
        )

    def run(self):
        # start loop for the CLI
        if self.stilts_model_only:
            self.stilts_model_loop()
        else:
            self.cli_loop()

    def _help(self):
        print("Example prompts:")
        print("1. ############")
        print(
            "Prompt: 'Create a command to match catalogues input.fits and input2.fits using RA and dec columns to within 1 arcsec'."
        )
        print("2. ############")
        print("Prompt: 'How can I convert from a fits file to a csv file?'")

    def eval_execute_command(self, command):
        """execute the command"""
        should_execute = input(f"\n\nDo you want to execute this? (y/n): ")
        if should_execute.lower() in ["yes", "y"]:
            returned_out = self.execute_command(command)
        else:
            print(f"{colors['green']}Command execution skipped.{colors['reset']}")
            returned_out = "Command execution skipped."

        return returned_out

    def execute_command(self, command):
        """execute the command using subprocess"""
        try:
            print(f"{colors['yellow']}")
            run = subprocess.run(
                command, shell=True, check=True, text=True, capture_output=True
            )
            print(f"{colors['reset']}")
            print(f"{colors['green']}Command executed successfully.{colors['reset']}")
            return run.stdout

        except subprocess.CalledProcessError as e:
            print(e.stderr)
            print(f"{colors['red']}Error executing command: {e}{colors['reset']}")
            return e.stderr


def main():
    parser = argparse.ArgumentParser(description="Stilts Agent CLI")
    parser.add_argument(
        "--inference_library",
        type=str,
        default="transformers",
        help="Inference library (transformers or llama_cpp)",
    )

    parser.add_argument(
        "--update",
        action="store_true",
        default=False,
        help="Update the STILTS models by re-downloading it from huggingface.",
    )

    parser.add_argument(
        "--num_proc", type=int, default=5, help="Number of processors for llama_cpp"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cpu or cuda or mps (mac))",
    )

    parser.add_argument(
        "--stilts_model_only",
        action="store_true",
        help="Run only the Stilts model for command generation",
    )

    parser.add_argument(
        "--precision_stilts_model",
        type=str,
        default="float16",
        help="Precision for Stilts model (float16, 8bit, or 4bit)",
    )

    parser.add_argument(
        "--precision_gen_model",
        type=str,
        default="8bit",
        help="Precision for Gen model (float16, 8bit, or 4bit)",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="float16",
        help="Precision for both models (float16, 8bit, or 4bit)",
    )

    args = parser.parse_args()

    # if --precision is set, use it for both models unless they are set individually
    if args.precision is not None:
        args.precision_stilts_model = args.precision
        args.precision_gen_model = args.precision

    cli = CLI(
        inference_library=args.inference_library,
        num_proc=args.num_proc,
        device=args.device,
        stilts_model_only=args.stilts_model_only,
        precision_stilts_model=args.precision_stilts_model,
        precision_gen_model=args.precision_gen_model,
        force_download=args.update,
    )
    cli.greating()
    cli.run()


if __name__ == "__main__":
    main()
