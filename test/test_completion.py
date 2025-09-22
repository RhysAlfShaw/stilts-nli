from stilts_nli.utils.completion import CombinedCompleter
import pytest
import os
from prompt_toolkit.document import Document
from prompt_toolkit.completion import CompleteEvent


@pytest.fixture
def temp_environment(tmpdir):
    """
    A pytest fixture to create a temporary testing environment.
    It creates a temporary directory with a few dummy files and a sub-directory,
    then changes the current working directory to it for the duration of a test.
    """
    original_cwd = os.getcwd()
    os.chdir(tmpdir)

    # create dummy files and a directory to test completion
    with open("test_file_1.txt", "w") as f:
        f.write("hello")
    with open("another-document.log", "w") as f:
        f.write("world")
    os.mkdir("test_dir")

    yield
    os.chdir(original_cwd)


def get_completion_texts(completer, text):
    """
    A helper function to simplify getting a sorted list of completion strings
    from a completer instance for a given input text.
    """
    document = Document(text, len(text))
    complete_event = CompleteEvent()
    completions = list(completer.get_completions(document, complete_event))
    return sorted([comp.text for comp in completions])


def test_standard_command_completion(temp_environment):
    """
    Tests that the completer suggests the full set of commands when not in quotes.
    """
    completer = CombinedCompleter()

    assert get_completion_texts(completer, "s") == ["save"]

    expected_commands = sorted(["exit", "quit", "help", "clear", "save"])
    assert get_completion_texts(completer, "") == expected_commands

    assert get_completion_texts(completer, "H") == ["help"]


def test_reduced_option_command_completion(temp_environment):
    """
    Tests that the completer suggests the reduced set of commands when initialized
    with `reduced_option=True`.
    """
    completer = CombinedCompleter(reduced_option=True)

    assert get_completion_texts(completer, "q") == sorted(["q", "quit"])

    assert get_completion_texts(completer, "save") == []

    expected_commands = sorted(["exit", "quit", "q"])
    assert get_completion_texts(completer, "") == expected_commands


def test_path_completion_inside_single_quotes(temp_environment):
    """
    Tests that the completer suggests file paths (and not commands or directories)
    when the cursor is inside single quotes.
    """
    completer = CombinedCompleter()

    text_in_quotes = "open 'a"
    expected_files = sorted(["another-document.log"])
    print(get_completion_texts(completer, text_in_quotes))
    assert get_completion_texts(completer, text_in_quotes) == expected_files

    text_with_partial = "open 'test_file"
    assert get_completion_texts(completer, text_with_partial) == ["test_file_1.txt"]


def test_path_completion_inside_double_quotes(temp_environment):
    """
    Tests that the completer suggests file paths (and not commands or directories)
    when the cursor is inside double quotes.
    """
    completer = CombinedCompleter()

    # text with an opening quote
    text_in_quotes = 'cat "a'
    expected_files = sorted(["another-document.log", "test_file_1.txt"])
    assert get_completion_texts(completer, text_in_quotes) == ["another-document.log"]


def test_completion_switches_back_after_quotes(temp_environment):
    """
    Tests that completion reverts to commands once a quoted string is closed.
    """
    completer = CombinedCompleter()

    # after a closed single-quoted string
    text = "command 'some_file.txt' && h"
    assert get_completion_texts(completer, text) == ["help"]

    # after a closed double-quoted string
    text = 'command "another_file.log" | s'
    assert get_completion_texts(completer, text) == ["save"]
