from prompt_toolkit.completion import (
    Completer,
    WordCompleter,
)
import os


class CombinedCompleter(Completer):
    """
    A prompt_toolkit completer that combines command and path completion.
    """

    def __init__(self, reduced_option=False):
        if reduced_option:
            COMMANDS = [
                "exit",
                "quit",
                "q",
            ]
        else:
            COMMANDS = [
                "exit",
                "quit",
                "help",
                "clear",
                "save",
            ]
        local_files = [
            f
            for f in os.listdir(os.getcwd())
            if os.path.isfile(os.path.join(os.getcwd(), f))
        ]
        self.command_completer = WordCompleter(COMMANDS, ignore_case=True)
        self.path_completer = WordCompleter(local_files, ignore_case=True)

    def get_completions(self, document, complete_event):
        """
        Yields completions based on the current context.
        """
        text_before_cursor = document.text_before_cursor

        # Check if the cursor is inside an odd number of single or double quotes
        in_single_quotes = text_before_cursor.count("'") % 2 == 1
        in_double_quotes = text_before_cursor.count('"') % 2 == 1

        if in_single_quotes or in_double_quotes:
            # If inside quotes, use the PathCompleter
            yield from self.path_completer.get_completions(document, complete_event)
        else:
            # Otherwise, use the WordCompleter for commands
            yield from self.command_completer.get_completions(document, complete_event)
