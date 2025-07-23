## main cli loop for stilts-agent.
import subprocess

from model import Model

class CLI:
    def __init__(self):
        self.model = Model()
    
    def cli_loop(self):
        while True:
            self.get_prompt()
            if self.prompt.lower() == 'exit' or self.prompt.lower() == 'quit' or self.prompt.lower() == 'q':
                print("Exiting CLI.")
                break

            command = self.model.generate_command(self.prompt)
            print(f"Generated STILTS command: {command}")

            self.eval_execute_command(command)
        

    def greating(self):
        print("Welcome to the Stilts CLI!")

    def get_prompt(self):
        """ask the user for a command"""
        print("What STILTS command would you like me to create for you? (if unsure, type 'help')")
        prompt_input = input("Enter a command: ")

        if prompt_input == 'help':
            self._help()
            return self.get_prompt()
        self.prompt = prompt_input   
    
    def run(self):
        # start loop for the CLI
        self.cli_loop()

    def _help(self):
        print("Example prompts:")
        print("1. ############")
        print("Prompt: 'Create a command to match catalogues input.fits and input2.fits using RA and dec columns to within 1 arcsec'.")
        print("2. ############")
        print("Prompt: ' How can I convert from a fits file to a csv file?'")

    def eval_execute_command(self, command):
        """execute the command"""
        should_execute = input(f"Do you want to execute the command: {command}? (y/n): ")
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