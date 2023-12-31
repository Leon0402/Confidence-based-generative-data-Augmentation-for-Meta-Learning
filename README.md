# Confidence-based-generative-data-Augmentation-for-Meta-Learning 

This project aims to develop a confidence-based generative data augmentation technique that can be applied in few-shot learning methods. 
The expected outcome is a method-agnostic data augmentation technique that improves classification accuracy in both within-domain and cross-domain scenarios. 

The code is originally taken from the [Cross-Domain MetaDL Competition](https://github.com/DustinCarrion/cd-metadl) and thus the code structure is influenced by the competition.

## Setup

Dependencies:
* Python 3.11 (pyenv can be used)
* Poetry ^1.7.0

Install dependencies from the lock file (virtual environment will be created automatically): 

```bash
poetry install 
```

> **_NOTE:_** For better compatibility with editors like vs code it might be helpful to run `poetry config virtualenvs.in-project true` beforehand. This will create the virtual environment in the project folder and allows the editor to autodetect it.

Open a shell within the virtual environment with: 

```bash
poetry shell
```

Alternatively prefix all following commands with `poetry run <command here>`. 


Download and verify the datasets:
```bash
python ./cdmetadl/helpers/initial_setup.py
```

Run the code with:

```bash
python -m cdmetadl.train \
    --config_path="configs/train/experiment.yml" \
    --model_dir=baselines/finetuning \
    --domain_type="cross-domain" \
    --overwrite_previous_results \
    --verbose 
```

Arguments can be adjusted as needed. Run `python -m cdmetadl.train --help` to get the documentation of the command line arguments. Logs will be written to the command line ouput as well as to tensorboard. 
To start tensorboard run: 

```bash
tensorboard --logdir .
```

Evaluation can be run with: 

```bash
python -m cdmetadl.eval \
    --training_output_dir="output/tmp/training/experiment/finetuning/cross-domain" \
    --config_path="configs/eval/experiment.yml" \
    --overwrite_previous_results \
    --verbose 
```

Open dashboard with: 

```bash
python -m cdmetadl.dashboard --eval-output-path "./output/tmp/eval"
```

## Automation Scripts 

For ease of use the following script can be used to run a model in cross-domain and with-domain setting: 

```bash
./scripts/train.sh --datasets_dir "./public_data" --model_dir "./baselines/finetuning"
```

```bash
./scripts/eval.sh --model_dir "./baselines/finetuning"
```