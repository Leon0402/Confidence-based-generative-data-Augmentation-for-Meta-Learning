# Confidence-based-generative-data-Augmentation-for-Meta-Learning 

This project aims to develop a confidence-based generative data augmentation technique that can be applied in few-shot learning methods. 
The expected outcome is a method-agnostic data augmentation technique that improves classification accuracy in both within-domain and cross-domain scenarios. 

The code is originally taken from the [Cross-Domain MetaDL Competition](https://github.com/DustinCarrion/cd-metadl) and thus the code structure is influenced by the competition.

## Setup

Dependencies:
* Python 3.11 (pyenv can be used)
* Poetry ^1.7.0

Install dependencies from the lock file in a virtual environment with: 
```py
$ poetry install 
```

> **_NOTE:_** For better compatibility with editors like vs code it might be helpful to run `poetry config virtualenvs.in-project true` beforehand. This will create the virtual environment in the project folder and allows the editor to autodetect it.

Open a shell within the virtual environment with: 
```
$ poetry shell
```

Alternatively prefix all following commands with `poetry run <command here>`. 

Run the code with: 
```py
python -m cdmetadl.run \
    --input_data_dir=public_data \
    --submission_dir=baselines/finetuning \
    --output_dir_ingestion=ingestion_output \
    --output_dir_scoring=scoring_output \
    --domain_type="cross-domain" \
    --verbose=True \
    --overwrite_previous_results=True \
    --test_tasks_per_dataset=10
```

Alternatively, you can also run `ingestion.py` for training and prediction and `scoring.py` for evaluation seperatly in a similar fashion.