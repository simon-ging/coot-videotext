# nntrainer Deep Learning Library

v0.2.6

## Features

**Background tasks** are taken care of: Fail-save checkpoint saving and loading, timing, logging, saving metrics, early stopping, profiling GPU and memory usage.

You can easily **create output tables** over all experiments or subsets:

~~~bash
python show_retrieval.py -g paper2020 --log_dir provided_experiments --mean --compact

# Output:
# experiment (num) |       v2p-r1|       p2v-r1|       c2s-r1|       s2c-r1|  time
# -----------------|-------------|-------------|-------------|-------------|----------
# anet_coot_run (3)|61.44% ±0.94%|61.56% ±0.82%| 0.00% ±0.00%| 0.00% ±0.00%|0.90 ±0.23
# yc2_100m_run (3) |75.35% ±2.67%|73.96% ±2.09%|15.47% ±0.04%|16.64% ±0.19%|0.20 ±0.02
# yc2_2d3d_run (3) |48.72% ±1.03%|47.63% ±1.42%| 5.53% ±0.17%| 5.97% ±0.21%|1.45 ±0.41
~~~

## Introduction

Experiments are organized hierarchically:

|         | Type                     | Group          | Name                 | Run                |
| ------- | ------------------------ | -------------- | -------------------- | ------------------ |
|         | A new type requires new boilerplate code. | Group experiment configurations, e.g. by dataset or by release. | The name of the configuration file you want to load. | A single training run of the configuration. |
| Example | `retrieval`              | `paper2020`    | `yc2_100m_coot`      | `run#`, `#=1,2,3`  |
| Command | `train_retrieval.py`     | `-g paper2020` | `-e yc2_100m_coot`   | `-r run -n 3`      |
| Config  | `config/retrieval/`      | `paper2020/`   | `yc2_100m_coot.yaml` |                    |
| Output  | `experiments/retrieval/` | `paper2020/`   | `yc2_100m_coot_`     | `run#/`, `#=1,2,3` |

Assuming you have setup datasets as described in the repository `README.md`, Train and show the results like this:

~~~bash
# train a model for 3 runs
python train_retrieval.py -g paper2020 -e yc2_100m_coot -r run -n 3
# show your results, average over the 3 runs
python show_retrieval.py -g paper2020 --mean
# show the provided results
python show_retrieval.py -g paper2020 --log_dir provided_experiments --mean --compact
~~~

Trainer, Model, Dataset are created based on the yaml config files.

### Minimum working example

The toy experiment in `nntrainer.examples.mlp_mnist`  shows how to set up a very simple experiment with this library. Run with:

~~~bash
python -m nntrainer.examples.run_mlp_mnist -e mnist
~~~

## Documentation

Overview of the training workflow:

1. `nntrainer.arguments` creates the console interface for the run script `train_retrieval.py`.
2. Experiments are defined as `yaml` files in the `config` folder.
3. This config is turned into a `dict` and the `same_as` fields are resolved (reference fields to avoid duplication in the configs).
4. The `dict` is then used to create a configuration object with class `src.config.config_retrieval.RetrievalConfig`.
5. `coot.dataset_retrieval` creates the dataset. `coot.model_retrieval` creates the model and does the forward pass. The model itself is in `nntrainer.models.transformer_legacy`, 4 instances of the `TransformerLegacy` class make up one full COOT model (visual/text and low/high level each). 
6. The training loop is in `coot.trainer_retrieval`, the background tasks (like checkpoint saving) happen in `nntrainer.trainer_base` and are called with the functions `self.hook_...` in the training loop. 

## Typing extensions

This library is statically typed. It uses the extensions in `nntrainer.typext`, [pydantic](https://pydantic-docs.helpmanual.io/) and the `typing` module. The goal is to provide good auto-complete and detect bugs early.

- `ConfigClass`: The input yaml config file is replicated with a class hierarchy of this type. See the configuration for retrieval training in  `coot.configs_retrieval.RetrievalConfig`.
- `ConstantHolder`: All string or other constants are saved in static classes of this type.
- `TypedNamedTuple` to define batches, datapoints, model outputs. Example in `coot.dataset_retrieval`
- `SaveableBaseModel` to save the training state separately from the trainer. 

## Advanced examples

Modify the entire configuration in-place with `-o` / `--config`. Fields are separated with commas and nested fields can be accessed with dots. In this example, the config is modified for debugging (smaller dataset, fewer epochs, smaller batches)

~~~bash
python train_retrieval.py -g paper2020 -e yc2_100m_coot -r test -n 1 --workers 0 -o train.batch_size=4,val.batch_size=4,dataset_train.max_datapoints=100,dataset_val.max_datapoints=100,train.num_epochs=2 --reset
~~~

**Output results:** The script `show_retrieval.py` arguments `-g`, `-s` supports searching experiments with the syntax of [gitignore files](https://git-scm.com/docs/gitignore). Note that on Linux you must escape strings containing stars with single quotes, on Windows you must not escape them. In the following example, the folder structure in `provided_experiments/retrieval/` is first matched against `paper2020/yc2*` to find the 2 YouCook2 experiments, then the experiment names are matched against `*100m*`, which will find 1 experiment.

~~~bash
python show_retrieval.py -g 'paper2020/yc2*' --log_dir provided_experiments -s '*100m*'
~~~

## Contributions

Issues and contributions are welcome. For issues, include information as described in the issue template. For contributions, read below.

### Design principles

- Separation of concerns: Dataset, Model and Trainer should be as independent of each other as possible.
- Check e.g. `nntrainer.models.poolers` on how a new torch `nn.Module` can be included in the library.

### Style Guide

- 120 characters per line.
- All variables should have type annotations, including function return types.
- There should be unit tests and integration tests for everything (incomplete).
- Docstrings for everything in [google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html):
    - All triple quotes must be on a line by themselves.
    - Argument and return types must be annotated if they exist.
    - Do not put any types into the docstring, they should be in the type annotations instead.
    - Describe the tensor shapes in the docstring.
    - For classes, do not annotate the `__init__` method, annotate `Args: ...` in the class docstring instead.
- Prepend any unused variables with `_`

### Tests

~~~bash
# Format your code correctly (e.g. with PyCharm). Test with:
pycodestyle --max-line-length 120 .

# Unit and integration tests must pass.
python -m pytest
python -m tests_nntrainer.integration_train
python -m tests_nntrainer.integration_deter

# Check test code coverage, v0.2.6 is 16%.
python -m pytest --cov coot --cov nntrainer .

# The pylinter should run at least without errors
pylint -s n nntrainer coot tests_nntrainer tests_coot --init-hook "import sys; sys.path.append('.')"

# Optional: MyPy static type check (will produce lots of errors and warnings)
mypy nntrainer coot tests_nntrainer tests_coot
~~~
