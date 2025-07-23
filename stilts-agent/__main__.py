## main cli loop for stilts-agent.

class CLI:
    def __init__(self):
        pass
    
    def cli_loop(self):
        while True:
            self.get_prompt()
            if self.prompt.lower() == 'exit' or self.prompt.lower() == 'quit' or self.prompt.lower() == 'q':
                print("Exiting CLI.")
                break
                
            # prompt the LLM.


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
    
    def llm(self):
        """call the llm to generate a command based on the prompt"""
        # This function would interact with the LLM to generate a command based on the prompt.
        pass

def main():
    cli = CLI()
    cli.greating()
    cli.run()

if __name__ == "__main__":
    main()