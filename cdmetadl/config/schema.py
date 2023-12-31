split_config = {
    "type": "object",
    "properties": {
        "train": {
            "type": "list"
        },
        "validation": {
            "type": "list"
        },
        "test": {
            "type": "list"
        },
    }
}

sampler_config = {
    "type": "object",
    "oneOf": [{
        "properties": {
            "value": {
                "type": "integer"
            }
        },
        "required": ["value"],
        "additionalProperties": False
    }, {
        "properties": {
            "range": {
                "type": "array",
                "items": {
                    "type": "integer"
                },
                "minItems": 2,
                "maxItems": 2
            }
        },
        "required": ["range"],
        "additionalProperties": False
    }, {
        "properties": {
            "choice": {
                "type": "array",
                "items": {
                    "type": "integer"
                }
            }
        },
        "required": ["choice"],
        "additionalProperties": False
    }]
}

generator_config = {
    "type": "object",
    "properties": {
        "batch_size": {
            "type": "integer"
        },
        "n_ways": sampler_config,
        "k_shots": sampler_config,
        "query_size": {
            "type": "integer"
        }
    },
    "required": ["n_ways", "k_shots", "query_size"]
}

train_dataset_schema = {
    "type": "object",
    "properties": {
        "split": {
            "cross-domain": split_config,
            "domain-independent": split_config
        },
        "train": generator_config,
        "validation": generator_config,
    },
    "required": ["split", "train", "test"]
}

model_schema = {
    "type": "object",
    "properties": {
        "number_of_batches": {
            "type": "integer"
        },
        "number_of_training_tasks": {
            "type": "integer"
        },
        "number_of_validation_tasks_per_dataset": {
            "type": "integer"
        },
        "validate_every": {
            "type": "integer"
        },
        "dropout_probability": {
            "type": "number"
        }
    },
    "required": [
        "number_of_batches", "number_of_training_tasks", "number_of_validation_tasks_per_dataset", "validate_every"
    ]
}

train_config_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "datset": train_dataset_schema,
        "model_schema": model_schema
    },
    "required": ["dataset", "model"]
}

eval_dataset_schema = {
    "type": "object",
    "properties": {
        "test": generator_config,
    },
    "required": ["test"]
}

eval_config_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "dataset": eval_dataset_schema
    },
    "required": []
}
