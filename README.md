# STILTS Natural Language Interface Commandline Tool.
[![Run tests](https://github.com/RhysAlfShaw/stilts-nli/actions/workflows/pytest.yml/badge.svg)](https://github.com/RhysAlfShaw/stilts-nli/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/RhysAlfShaw/stilts-nli/graph/badge.svg?token=E1FQ94JAHN)](https://codecov.io/gh/RhysAlfShaw/stilts-nli)
[![PyPI version](https://badge.fury.io/py/stilts_nli.svg)](https://badge.fury.io/py/stilts_nli)
## Dependencies.
[STILTS](https://www.star.bris.ac.uk/~mbt/stilts/ "https://www.star.bris.ac.uk/~mbt/stilts/") (accesable via the command stilts in the terminal.) This only limits the execution of the generated command.
## Installation

STILTS-NLI can be installed via pip with atleast python 3.12. It is recomended this is done in a virtual environment or conda environment.

```bash
pip install stilts-nli
```
We can then run the command.
```bash
stilts-nli
```

When first starting up the tool. It will download the models. This may take a few minutes.

### Apple Silicon GPU (mps)

Transformers has support for apple silicon's mps device, allowing for inference on apples GPU. This has joint memory with the CPU so becarful you have enough memory. The STILTS only model is executable, suing this command:

```bash
stilts-nli --device mps --stilts_only_model
```

### Manual installation (conda)

Alternatively it can be installed manually through conda.

```bash
git clone git@github.com:RhysAlfShaw/stilts-nli.git
cd stilts-nli
```

Using anaconda, create and activate the stilts-agent environment:

```bash
conda env create -f environment.yml
conda activate stilts-agent
```

Test with this worked:

```bash
python stilts-nli
```

## Using quantised models with transformers

To use the 8bit and 4bit models with transformers then you also need to install bitsandbytes. This package required a cuda device, so if you have limited memory use llama_cpp_python for inference on use only the stilts model.

## Installing Llama_cpp_python

If you run into trouble installing `llama_cpp_python` and are on a linux machine it is likely a problem with C compilers. A work around is installing (with conda and also installing packages `gcc` and `gxx`).

## Docker 

If you dont want to or cannot install locally then use the local dockerfile to use the tool. You can do this by
cloning this directory.
```
git clone https://github.com/RhysAlfShaw/stilts-nli.git
cd stilts-nli
```
Now build the container

```bash
docker compose up -d
```
Lets go into the running container and lanch stilts-nli

```bash
docker exec -it stilts-nli /bin/bash -c "stilts-nli"
```
When you are finished rememeber to stop the container.

```
docker compose down
```

## Examples

To start STILTS-NLI with default f16 precision with a Cuda device.

```bash
stilts-nli
```

To start STILTS-NLI with 5 cores f16 precision on a CPU using llama_cpp for inference.

```bash
stilts-nli --num_proc 5 --device cpu --inference_library llama_cpp
```

To just ask directly to the finetuned model on the GPU.

```bash
stilts-nli --stilts_model_only 
```

Here is an example of a conversation with STILTS-NLI.

![STILTS-NLI Example](docs/example.png "STILTS-NLI Example")

## Usage

On startup different arguments can be parsed to change which models will be used and what resources you
want to allocate to the program. These include:

* --inference_library. This allows the user to choose which of the supported inference libraries to use
(transformers or llama.cpp).
* --num_proc. The number of CPUs you want any CPU inference to use.
* --device. The device to run inference on (CPU or cuda GPU).
* --stilts_model_only. Including this will run only the direct STILTSgen model. This saves on compu-
tation.
* --precision_stilts_model. The precision of the STILTSgen model, useful for saving memory at the
cost of reduced performance (f16, 8bit and 4bit).
* --precision_gen_model. The precision of the general chat model, useful for saving memory at the cost
of preformance (f16, 8bit and 4bit).
* --precision. The precision for both model this overrides the individual set precisions (f16, 8bit, 4bit).


Once all models have loaded successfully, you will be greeted with the following.
```bash
Welcome to the Stilts Natural Language Interface!
This tool allows you to generate STILTS commands and execute them using a
natural language. You can ask the model to create commands based on your
prompts. Once it generates a command ask it to execute it.
Type ’help/h’ for guidence, ’clear/c’ to clear the message history,
’quit/q’ to exit. Save message history to a file type ’save/s’.
Welcome to the Stilts CLI!
>> |
```

From here you are now interacting with the general model, view the generated stilts command, and do any of
the additional options. These are asking for help (examples on how to use the tool). Clear the message history
so you can start again. Quit the sessions and return to the terminal. And save the message history.
When a stilts command is generated the model is also prompted to explain what it is doing.

```bash
>> Can you make me an example stilts command that will match two catalogues based on the
column value OBJECT_ID?

stilts tmatch2 in1=catalogue1.fits in2=catalogue2.fits out=id_match.fits matcher=exact
values1=OBJECT_ID values2=OBJECT_ID

This command performs a simple object matching operation. It reads two FITS tables,
‘catalogue1.fits‘ and ‘catalogue2.fits‘, and uses the ‘match2‘ algorithm to find
all objects from the first catalogue that have an exact match in the ’ID’ column
from the second catalogue. The result is a new FITS table containing only the ID
columns from both the input tables, named ‘ID_match.fits‘.
```

## Compute Requirements.

Low precision versions of both models are (or will be) avalible which will allow inference on low memory GPUs or lower spec laptops. These are:
STILTSgen model.
- f16 (~6Gb)
- 8bit (~3.3GB)
- 4bit (~2.7GB)

Gen model (Llama-3.2-3B-Instruct):
- f16 (~12GB)
- 8bit (~3.5GB)
- 4bit (~3GB)

For full precsion of both models you require a minimum of ~18GB of GPU or CPU RAM.