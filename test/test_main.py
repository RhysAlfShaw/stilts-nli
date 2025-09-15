import pytest
import subprocess

from unittest.mock import MagicMock, call, mock_open
from stilts_nli.__main__ import CLI


@pytest.fixture
def cli_instance():
    """
    Provides a CLI instance with mocked StiltsModel and GenModel.
    """
    cli = CLI(
        inference_library="transformers",
        num_proc=4,
        device="cpu",
        stilts_model_only=False,
        test_mode=True,
    )

    return cli


param_names = [
    "inference_library",
    "num_proc",
    "device",
    "stilts_model_only",
    "precision_stilts_model",
    "precision_gen_model",
]
test_cases = [
    ("llama_cpp", 1, "cuda", True, "4bit", "8bit"),
    ("transformers", 1, "cuda", False, "4bit", "8bit"),
    ("llama_cpp", 1, "cpu", True, "4bit", "8bit"),
    ("transformers", 1, "cpu", False, "4bit", "8bit"),
    ("llama_cpp", 5, "cuda", True, "4bit", "8bit"),
    ("transformers", 5, "cuda", False, "4bit", "8bit"),
    ("llama_cpp", 5, "cpu", True, "4bit", "float16"),
    ("transformers", 5, "cuda", False, "float16", "float16"),
]


@pytest.mark.parametrize(
    param_names,
    test_cases,
)
def test_cli_initialization(
    inference_library,
    num_proc,
    device,
    stilts_model_only,
    precision_stilts_model,
    precision_gen_model,
):
    """
    Tests if the CLI class initializes correctly and sets up its models.
    """

    cli = CLI(
        inference_library=inference_library,
        num_proc=num_proc,
        device=device,
        stilts_model_only=stilts_model_only,
        precision_stilts_model=precision_stilts_model,
        precision_gen_model=precision_gen_model,
        force_download=False,
        test_mode=True,
    )

    assert cli.inference_library == inference_library
    assert cli.num_proc == num_proc
    assert cli.device == device
    assert cli.stilts_model_only == stilts_model_only
    assert cli.precision_stilts_model == precision_stilts_model
    assert cli.precision_gen_model == precision_gen_model


def test_add_to_message_history(cli_instance):
    """
    Tests the simple addition of a message to the history.
    """
    cli = cli_instance
    assert len(cli.message_history) == 0

    test_message = {"role": "user", "content": "hello"}
    cli.add_to_message_history(test_message)

    assert len(cli.message_history) == 1
    assert cli.message_history[0] == test_message


def test_help_command(cli_instance, capsys):
    """
    Tests if the _help method prints the expected guidance.
    """
    cli = cli_instance
    cli._help()
    captured = capsys.readouterr()
    assert "Example prompts:" in captured.out
    assert "Create a command to match catalogues" in captured.out


def test_execute_command_success(cli_instance, mocker):
    """
    Tests successful command execution using a mocked subprocess.
    """
    cli = cli_instance

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.stdout = "Success output"
    mock_run.return_value.stderr = ""

    command = "ls -l"
    result = cli.execute_command(command)

    mock_run.assert_called_once_with(
        command, shell=True, check=True, text=True, capture_output=True
    )
    assert result == "Success output"


def test_execute_command_failure(cli_instance, mocker):
    """
    Tests failed command execution using a mocked subprocess.
    """
    cli = cli_instance

    mock_run = mocker.patch("subprocess.run")
    error = subprocess.CalledProcessError(1, cmd="bad_command", stderr="Error output")
    mock_run.side_effect = error

    command = "bad_command"
    result = cli.execute_command(command)

    mock_run.assert_called_once_with(
        command, shell=True, check=True, text=True, capture_output=True
    )
    assert result == "Error output"


def test_eval_execute_command_user_accepts(cli_instance, mocker):
    """
    Tests the evaluation wrapper where the user types 'y' to execute.
    """
    cli = cli_instance
    mocker.patch("builtins.input", return_value="y")
    mock_execute = mocker.patch.object(cli, "execute_command", return_value="Executed")

    command_to_run = "stilts tpipe ..."
    result = cli.eval_execute_command(command_to_run)

    mock_execute.assert_called_once_with(command_to_run)
    assert result == "Executed"


def test_eval_execute_command_user_rejects(cli_instance, mocker):
    """
    Tests the evaluation wrapper where the user types 'n' to skip.
    """
    cli = cli_instance
    mocker.patch("builtins.input", return_value="n")
    mock_execute = mocker.patch.object(cli, "execute_command")

    result = cli.eval_execute_command("stilts tpipe ...")

    mock_execute.assert_not_called()
    assert result == "Command execution skipped."
