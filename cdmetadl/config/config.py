__all__ = ["DataFormat", "DatasetConfig", "ModelConfig", "read_config"]

from dataclasses import dataclass
from enum import Enum
import pathlib
import yaml

import jsonschema

import cdmetadl.samplers

from .schema import config_schema


class DataFormat(str, Enum):
    BATCH = "batch"
    TASK = "task"

    def __str__(self):
        return self.value


@dataclass
class DatasetConfig:
    batch_size: int
    n_ways: cdmetadl.samplers.Sampler
    k_shots: cdmetadl.samplers.Sampler
    query_size: int

    @staticmethod
    def from_json(json_config: dict) -> "DatasetConfig":

        def parse_sampler(sampler_config: dict):
            if "value" in sampler_config:
                return cdmetadl.samplers.ValueSampler(sampler_config['value'])
            if "range" in sampler_config:
                return cdmetadl.samplers.RangeSampler(*sampler_config['range'])
            if "choice" in sampler_config:
                return cdmetadl.samplers.ChoiceSampler(sampler_config['choice'])

        return DatasetConfig(
            batch_size=json_config.get('batch_size', None), n_ways=parse_sampler(json_config['n_ways']),
            k_shots=parse_sampler(json_config['k_shots']), query_size=json_config['query_size']
        )


@dataclass
class ModelConfig:
    number_of_batches: int
    number_of_training_tasks: int
    number_of_validation_tasks_per_dataset: int
    validate_every: int

    @staticmethod
    def from_json(json_config: dict) -> "ModelConfig":
        return ModelConfig(
            number_of_batches=json_config['number_of_batches'],
            number_of_training_tasks=json_config['number_of_training_tasks'],
            number_of_validation_tasks_per_dataset=json_config['number_of_validation_tasks_per_dataset'],
            validate_every=json_config['validate_every']
        )


def read_config(path: pathlib.Path) -> dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    jsonschema.validate(instance=config, schema=config_schema)
    return config
