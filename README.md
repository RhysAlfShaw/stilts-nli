# STILTS Natural Language Interface Commandline Tool.

## Dependencies.
- STILTS (accesable via the command stilts in the terminal.) [STILTS](https://www.star.bris.ac.uk/~mbt/stilts/ "https://www.star.bris.ac.uk/~mbt/stilts/")

## Installation

```bash
conda env create -f environment.yml
conda activate stilts-agent
```

Test with:

```
python stilts-agent
```


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