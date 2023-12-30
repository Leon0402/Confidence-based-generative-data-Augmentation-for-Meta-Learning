config_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "dataset": {
            "type": "object",
            "properties": {
                "batch_size": {
                    "type": "integer"
                },
                "n_ways": {
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
                },
                "k_shots": {
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
                },
                "query_size": {
                    "type": "integer"
                }
            },
            "required": ["batch_size", "n_ways", "k_shots", "query_size"]
        },
        "model": {
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
                }
            },
            "required": [
                "number_of_batches", "number_of_training_tasks", "number_of_validation_tasks_per_dataset",
                "validate_every"
            ]
        }
    },
    "required": ["dataset", "model"]
}
