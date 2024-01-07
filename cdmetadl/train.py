import argparse
import datetime
import pathlib
import random
import pickle
import shutil
from enum import Enum

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers
import cdmetadl.logger


class DomainType(Enum):
    CROSS_DOMAIN = "cross-domain"
    WITHIN_DOMAIN = "within-domain"

    def __str__(self):
        return self.value


class DataFormat(Enum):
    BATCH = "batch"
    TASK = "task"

    def __str__(self):
        return self.value


def read_generator_configs(config_file: pathlib.Path) -> tuple[str, dict, dict, dict]:
    train_data_format = DataFormat.TASK

    train_generator_config = {
        "N": 5,
        "min_N": None,
        "max_N": None,
        "k": None,
        "min_k": 1,
        "max_k": 20,
        "query_images_per_class": 20,
        "batch_size": 16
    }

    valid_generator_config = {
        "N": None,
        "min_N": 2,
        "max_N": 5,
        "k": None,
        "min_k": 1,
        "max_k": 20,
        "query_images_per_class": 20
    }

    test_generator_config = {
        "N": None,
        "min_N": 2,
        "max_N": 5,
        "k": None,
        "min_k": 1,
        "max_k": 20,
        "query_images_per_class": 20
    }

    if config_file.exists():
        user_config = cdmetadl.helpers.general_helpers.load_json(config_file)
        if "train_data_format" in user_config:
            train_data_format = DataFormat(user_config["train_data_format"])
        if "train_config" in user_config:
            train_generator_config.update(user_config["train_config"])
        if "valid_config" in user_config:
            valid_generator_config.update(user_config["valid_config"])

    return train_data_format, train_generator_config, valid_generator_config, test_generator_config


def define_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--seed', type=int, default=42, help='Seed to make results reproducible. Default: 42.')
    parser.add_argument(
        '--verbose', action=argparse.BooleanOptionalAction, default=True,
        help='Show various progression messages. Default: True.'
    )
    parser.add_argument(
        '--model_dir', type=pathlib.Path, required=True, help='Path to the directory containing the solution to use.'
    )
    parser.add_argument(
        '--datasets', nargs='*', help=
        'Specify the datasets that will be used for. Default: Uses all datasets in the cross domain setup or one random selected dataset in the within domain setup.'
    )
    parser.add_argument(
        '--domain_type', type=DomainType, choices=list(DomainType), default=DomainType.CROSS_DOMAIN, help=
        "Choose the domain type for meta-learning. 'cross-domain' indicates a setup where multiple distinct datasets are utilized, ensuring that the training, validation, and testing sets are entirely separate, thereby promoting generalization across diverse data sources. 'within-domain', on the other hand, refers to using a single dataset, segmented into different classes for training, validation, and testing, focusing on learning variations within a more homogeneous data environment. Default: cross-domain."
    )
    parser.add_argument(
        '--data_dir', type=pathlib.Path, default='./public_data',
        help='Default location of the directory containing the meta_train and meta_test data. Default: "./public_data".'
    )
    parser.add_argument(
        "--train_split", type=float, default=0.6, help=
        "Proportion of dataset used for training. Train, validation and test split should sum up to one. Default: 0.6."
    )
    parser.add_argument(
        "--validation_split", type=float, default=0.2, help=
        "Proportion of dataset used for validation. Train, validation and test split should sum up to one. Default: 0.2."
    )
    parser.add_argument(
        "--test_split", type=float, default=0.2, help=
        "Proportion of dataset used for testing. Train, validation and test split should sum up to one. Default: 0.2."
    )
    parser.add_argument(
        '--output_dir', type=pathlib.Path, default='./training_output',
        help='Path to the output directory for training. Default: "./training_output".'
    )
    parser.add_argument(
        '--overwrite_previous_results', action=argparse.BooleanOptionalAction, default=False,
        help='Overwrites the previous output directory instead of renaming it. Default: False.'
    )
    parser.add_argument(
        '--pseudo_DA', action=argparse.BooleanOptionalAction, default=False,
        help='Uses pseudo data augmentation. Default: False.'
    )
    return parser


def process_args(args: argparse.Namespace) -> None:
    args.data_dir = args.data_dir.resolve()
    args.model_dir = args.model_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.datasets:
        args.datasets = [dir.name for dir in args.data_dir.iterdir() if dir.is_dir() and dir.name.isupper()]
        if args.domain_type == DomainType.WITHIN_DOMAIN:
            args.datasets = [random.choice(args.datasets)]

    if args.domain_type == 'within-domain' and len(args.datasets) > 1:
        raise ValueError("More than one dataset specified for within-domain scenario")


def prepare_directories(args: argparse.Namespace) -> None:
    cdmetadl.helpers.general_helpers.exist_dir(args.data_dir)
    cdmetadl.helpers.general_helpers.exist_dir(args.model_dir)

    args.output_dir /= f"{args.model_dir.name}/{args.domain_type}/dataset"

    if args.output_dir.exists() and args.overwrite_previous_results:
        shutil.rmtree(args.output_dir)
    elif args.output_dir.exists():
        timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        args.output_dir.rename(args.output_dir.parent / f"{args.output_dir.name}_{timestamp}")
    args.output_dir.mkdir(parents=True)

    args.logger_dir = args.output_dir / "logs"
    args.logger_dir.mkdir()

    args.output_model_dir = args.output_dir / "model"
    args.output_model_dir.mkdir()

    args.dataset_output_dir = args.output_dir / "datasets"
    args.dataset_output_dir.mkdir()


def prepare_data_generators(
    args: argparse.Namespace
) -> tuple[cdmetadl.dataset.DataGenerator, cdmetadl.dataset.TaskGenerator]:
    datasets_info = cdmetadl.helpers.general_helpers.check_datasets(args.data_dir, args.datasets, args.verbose)
    meta_dataset = cdmetadl.dataset.MetaImageDataset([
        cdmetadl.dataset.ImageDataset(name, info) for name, info in datasets_info.items()
    ])

    
    splitting = [args.train_split, args.validation_split, args.test_split]
    match args.domain_type:
        case DomainType.CROSS_DOMAIN:
            train_dataset, val_dataset, test_dataset = cdmetadl.dataset.random_meta_split(
                meta_dataset, splitting, seed=args.seed
            )
        case DomainType.WITHIN_DOMAIN:
            train_dataset, val_dataset, test_dataset = cdmetadl.dataset.random_class_split(
                meta_dataset, splitting, seed=args.seed
            )

    with open(args.dataset_output_dir / 'train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(args.dataset_output_dir / 'val_dataset.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)
    with open(args.dataset_output_dir / 'test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)

    train_data_format, train_generator_config, valid_generator_config, _ = read_generator_configs(
        args.model_dir / "config.json"
    )

    match train_data_format:
        case DataFormat.TASK:
            meta_train_generator = cdmetadl.dataset.TaskGenerator(train_dataset, train_generator_config)
        case DataFormat.BATCH:
            meta_train_generator = cdmetadl.dataset.BatchGenerator(train_dataset)

    meta_val_generator = cdmetadl.dataset.TaskGenerator(val_dataset, valid_generator_config, sample_dataset=True)
    return meta_train_generator, meta_val_generator


def meta_learn(
    args: argparse.Namespace, meta_train_generator: cdmetadl.dataset.DataGenerator,
    meta_val_generator: cdmetadl.dataset.TaskGenerator
) -> None:
    model_module = cdmetadl.helpers.general_helpers.load_module_from_path(args.model_dir / "model.py")

    logger = cdmetadl.logger.Logger(args.logger_dir)
    meta_learner = model_module.MyMetaLearner(
        meta_train_generator.number_of_classes, meta_train_generator.total_number_of_classes, logger
    )
    meta_learner.meta_fit(meta_train_generator, meta_val_generator).save(args.output_model_dir)


def main():
    parser = define_argparser()
    args = parser.parse_args()

    cdmetadl.helpers.general_helpers.set_seed(args.seed)

    cdmetadl.helpers.general_helpers.vprint("\nProcess command line arguments", args.verbose)
    process_args(args)
    cdmetadl.helpers.general_helpers.vprint("[+] Command line arguments processed", args.verbose)

    cdmetadl.helpers.general_helpers.vprint("\nPrepare directories", args.verbose)
    prepare_directories(args)
    cdmetadl.helpers.general_helpers.vprint("[+] Prepared directories", args.verbose)

    cdmetadl.helpers.general_helpers.vprint("\nPreparing dataset", args.verbose)
    meta_train_generator, meta_val_generator = prepare_data_generators(args)
    cdmetadl.helpers.general_helpers.vprint("[+]Datasets prepared", args.verbose)

    cdmetadl.helpers.general_helpers.vprint("\nMeta-training your meta-learner...", args.verbose)
    meta_learn(args, meta_train_generator, meta_val_generator)
    cdmetadl.helpers.general_helpers.vprint("[+] Meta-learner meta-trained", args.verbose)

    # Rename file based on test dataset. TODO: Little bit hacky
    with open(args.dataset_output_dir / 'test_dataset.pkl', 'rb') as f:
        test_dataset: cdmetadl.dataset.MetaImageDataset = pickle.load(f)

    final_output_path = args.output_dir.parent / "-".join([dataset.name for dataset in test_dataset.datasets])
    final_output_path.mkdir(parents=True)
    args.output_dir.rename(final_output_path)


if __name__ == "__main__":
    main()
