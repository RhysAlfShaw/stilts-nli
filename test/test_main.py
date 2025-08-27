import pytest
import subprocess
import json

from unittest.mock import MagicMock, call, mock_open

from stilts_agent.__main__ import CLI


@pytest.fixture
def cli_instance(mocker):
    """
    Provides a CLI instance with mocked StiltsModel and GenModel.
    """
    # mock the classes themselves, so when CLI.__init__ is called,
    # it instantiates our mocks instead of the real classes.
    mock_stilts_model_class = mocker.patch("stilts_agent.stilts_model.StiltsModel")
    mock_gen_model_class = mocker.patch("stilts_agent.gen_model.GenModel")

    # create instances of the mocks that will be returned by the class constructors
    mock_stilts_instance = MagicMock()
    mock_gen_instance = MagicMock()
    mock_stilts_model_class.return_value = mock_stilts_instance
    mock_gen_model_class.return_value = mock_gen_instance

    cli = CLI(
        inference_library="transformers",
        num_proc=4,
        device="cpu",
        stilts_model_only=False,
    )

    # return the cli and the mock instances so we can control and inspect them in tests
    return cli, mock_stilts_instance, mock_gen_instance


def test_cli_initialization(mocker):
    """
    Tests if the CLI class initializes correctly and sets up its models.
    """
    mock_stilts = mocker.patch("stilts_agent.stilts_model.StiltsModel")
    mock_gen = mocker.patch("stilts_agent.gen_model.GenModel")

    cli = CLI(
        inference_library="llama_cpp",
        num_proc=8,
        device="cuda",
        stilts_model_only=False,
        precision_stilts_model="4bit",
        precision_gen_model="8bit",
    )

    assert cli.inference_library == "llama_cpp"
    assert cli.num_proc == 8
    assert cli.device == "cuda"
    assert not cli.stilts_model_only

    # check if models were instantiated with the correct parameters
    mock_stilts.assert_called_once_with(
        inference_library="test_lib", num_proc=8, device="cuda", precision="4bit"
    )
    mock_gen.assert_called_once_with(
        inference_library="test_lib", num_proc=8, device="cuda", precision="8bit"
    )


def test_add_to_message_history(cli_instance):
    """
    Tests the simple addition of a message to the history.
    """
    cli, _, _ = cli_instance
    assert len(cli.message_history) == 0

    test_message = {"role": "user", "content": "hello"}
    cli.add_to_message_history(test_message)

    assert len(cli.message_history) == 1
    assert cli.message_history[0] == test_message


def test_help_command(cli_instance, capsys):
    """
    Tests if the _help method prints the expected guidance.
    """
    cli, _, _ = cli_instance
    cli._help()
    captured = capsys.readouterr()
    assert "Example prompts:" in captured.out
    assert "Create a command to match catalogues" in captured.out


def test_execute_command_success(cli_instance, mocker):
    """
    Tests successful command execution using a mocked subprocess.
    """
    cli, _, _ = cli_instance

    # Mock subprocess.run to simulate a successful command
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
    cli, _, _ = cli_instance

    # Mock subprocess.run to raise a CalledProcessError
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
    cli, _, _ = cli_instance
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
    cli, _, _ = cli_instance
    mocker.patch("builtins.input", return_value="n")
    mock_execute = mocker.patch.object(cli, "execute_command")

    result = cli.eval_execute_command("stilts tpipe ...")

    mock_execute.assert_not_called()
    assert result == "Command execution skipped."


# --- More Complex Loop Tests ---


def test_cli_loop_clear_command(cli_instance, mocker):
    """
    Tests if the 'clear' command clears message history within the main loop.
    """
    cli, _, _ = cli_instance

    # Simulate user typing 'clear' then 'quit'
    mocker.patch.object(cli, "get_input", side_effect=["clear", "quit"])

    cli.message_history = [{"role": "user", "content": "some previous message"}]
    assert len(cli.message_history) == 1

    cli.cli_loop()

    assert len(cli.message_history) == 0


def test_cli_loop_save_command(cli_instance, mocker):
    """
    Tests if the 'save' command correctly saves the history to a file.
    """
    cli, _, _ = cli_instance

    # Mock prompt for filename and then quit
    mocker.patch(
        "stilts_agent_cli.prompt_session_history.prompt",
        side_effect=[
            "my_history_file",
            "save",
            "quit",
        ],  # User types 'save', then filename, then 'quit'
    )
    # This mock is a bit tricky. We need to mock get_input to get into the save branch,
    # and then prompt_session_history.prompt to get the filename.
    # A simpler way for this test is to mock get_input to return 'save' then 'quit'
    # and the prompt toolkit prompt to just return the filename.
    mocker.patch.object(cli, "get_input", side_effect=["save", "quit"])
    mocker.patch(
        "stilts_agent_cli.prompt_session_history.prompt", return_value="my_history"
    )

    # Mock the open() function to avoid creating a real file
    m = mock_open()
    mocker.patch("builtins.open", m)

    cli.message_history = [{"role": "user", "content": "save this"}]
    cli.cli_loop()

    m.assert_called_once_with("my_history.json", "w")
    handle = m()

    # Check what was written to the file
    written_data = "".join(call.args[0] for call in handle.write.call_args_list)
    assert json.loads(written_data) == cli.message_history


def test_cli_loop_stilts_command_generation_call(cli_instance, mocker):
    """
    Tests the full flow of generating a STILTS command from a tool call.
    """
    cli, mock_stilts, mock_gen = cli_instance

    user_prompt = "generate a crossmatch command"
    gen_model_response = '[tool_code]\nprint(stilts_command_generation(description="crossmatch two tables"))\n[/tool_code]'
    stilts_command = "stilts tmatch2 ..."
    stilts_explanation = "This command performs a crossmatch."

    mocker.patch.object(cli, "get_input", side_effect=[user_prompt, "quit"])

    # Configure mock responses
    mock_gen.generate_stream.return_value = iter([gen_model_response])
    mock_stilts.generate_stream.side_effect = [
        iter([stilts_command]),  # First call generates the command
        iter([stilts_explanation]),  # Second call generates the explanation
    ]

    cli.cli_loop()

    # Verify GenModel was called with the user prompt
    mock_gen.generate_stream.assert_called_once()
    assert mock_gen.generate_stream.call_args[0][0][-1]["content"] == user_prompt

    # Verify StiltsModel was called correctly twice
    expected_calls = [
        call("crossmatch two tables"),
        call(f"Explain the following STILTS command: {stilts_command}"),
    ]
    mock_stilts.generate_stream.assert_has_calls(expected_calls)

    # Check that the final message history is correct
    assert cli.message_history[-1]["role"] == "assistant"
    assert stilts_command in cli.message_history[-1]["content"]
    assert stilts_explanation in cli.message_history[-1]["content"]


def test_cli_loop_execute_stilts_command_call(cli_instance, mocker):
    """
    Tests the full flow of executing a STILTS command from a tool call.
    """
    cli, _, mock_gen = cli_instance

    user_prompt = "now execute it"
    command_to_execute = "stilts plot ..."
    gen_model_response = f'[tool_code]\nprint(execute_stilts_command(stilts_command="{command_to_execute}"))\n[/tool_code]'
    execution_output = "Plot generated successfully."

    mocker.patch.object(cli, "get_input", side_effect=[user_prompt, "quit"])
    mocker.patch.object(cli, "eval_execute_command", return_value=execution_output)

    mock_gen.generate_stream.return_value = iter([gen_model_response])

    cli.cli_loop()

    # Verify GenModel was called with user prompt
    mock_gen.generate_stream.assert_called_once()

    # Verify our eval/execute method was called with the correct command
    cli.eval_execute_command.assert_called_once_with(command_to_execute)

    # Check that the python/tool output was added to history
    assert cli.message_history[-1] == {"role": "python", "content": execution_output}
