import argparse
import yaml
import pathlib
import random
import pickle
import shutil
from enum import Enum

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers
import cdmetadl.logger
import cdmetadl.config


class DomainType(Enum):
    WITHIN_DOMAIN = "within-domain"
    CROSS_DOMAIN = "cross-domain"
    DOMAIN_INDEPENDENT = "domain-independent"

    def __str__(self):
        return self.value


def define_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--seed', type=int, default=42, help='Seed to make results reproducible. Default: 42.')
    parser.add_argument(
        '--verbose', action=argparse.BooleanOptionalAction, default=True,
        help='Show various progression messages. Default: True.'
    )
    parser.add_argument(
        '--model_dir', type=pathlib.Path, required=True, help='Path to the directory containing the model to use.'
    )
    parser.add_argument('--config_path', type=pathlib.Path, required=True, help='Path to config to use.')
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
        "--train_split", type=float, default=0.5, help=
        "Proportion of dataset used for training. Train, validation and test split should sum up to one. Default: 0.5."
    )
    parser.add_argument(
        "--validation_split", type=float, default=0.25, help=
        "Proportion of dataset used for validation. Train, validation and test split should sum up to one. Default: 0.25."
    )
    parser.add_argument(
        "--test_split", type=float, default=0.25, help=
        "Proportion of dataset used for testing. Train, validation and test split should sum up to one. Default: 0.25."
    )
    parser.add_argument(
        '--output_dir', type=pathlib.Path, default='./output/tmp/training',
        help='Path to the output directory for training. Default: "./output/tmp/training".'
    )
    parser.add_argument(
        '--overwrite_previous_results', action=argparse.BooleanOptionalAction, default=False,
        help='Overwrites the previous output directory instead of renaming it. Default: False.'
    )
    parser.add_argument(
        '--use_tensorboard', action=argparse.BooleanOptionalAction, default=True,
        help='Specify if you want to create logs fo the tensorboard for the current run. Default: True.'
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

    if args.domain_type == DomainType.WITHIN_DOMAIN and len(args.datasets) > 1:
        raise ValueError("More than one dataset specified for within-domain scenario")

    args.config = cdmetadl.config.read_train_config(args.config_path)
    # Overwrite config values from given arguments
    args.config["model"]["path"] = str(args.model_dir.relative_to(pathlib.Path.cwd()))


def prepare_directories(args: argparse.Namespace) -> None:
    cdmetadl.helpers.general_helpers.exist_dir(args.data_dir)
    cdmetadl.helpers.general_helpers.exist_dir(args.model_dir)

    args.output_dir /= f"{args.config_path.stem}/{args.model_dir.name}/{args.domain_type}"
    if args.domain_type == DomainType.WITHIN_DOMAIN:
        args.output_dir /= "".join(args.datasets)

    if args.output_dir.exists() and args.overwrite_previous_results:
        shutil.rmtree(args.output_dir)
    elif args.output_dir.exists():
        raise ValueError(
            f"Output directory {args.output_dir} exists and overwrite previous results option is not provided"
        )
    args.output_dir.mkdir(parents=True)

    args.logger_dir = args.output_dir / "logs"
    args.logger_dir.mkdir()

    args.output_model_dir = args.output_dir / "model"
    args.output_model_dir.mkdir()

    args.dataset_output_dir = args.output_dir / "datasets"
    args.dataset_output_dir.mkdir()

    if args.use_tensorboard:
        args.tensorboard_output_dir = args.output_dir / "tensorboard"
        args.tensorboard_output_dir.mkdir()


def prepare_data_generators(
    args: argparse.Namespace
) -> tuple[cdmetadl.dataset.DataGenerator, cdmetadl.dataset.TaskGenerator]:
    datasets_info = cdmetadl.helpers.general_helpers.check_datasets(args.data_dir, args.datasets, args.verbose)
    meta_dataset = cdmetadl.dataset.MetaImageDataset([
        cdmetadl.dataset.ImageDataset(name, info) for name, info in datasets_info.items()
    ])

    # TODO: Fix random cross domain split
    # TODO: Add random domain independent splitting
    # TODO: Allow random splitting and overwrite config before writing it out
    match args.domain_type:
        case DomainType.DOMAIN_INDEPENDENT:
            split_config = args.config["dataset"]["split"]["domain_independent"]
            splitting = [split_config["train"], split_config["validation"], split_config["test"]]
            train_dataset, val_dataset, test_dataset = cdmetadl.dataset.split_by_names(meta_dataset, splitting)
        case DomainType.CROSS_DOMAIN:
            split_config = args.config["dataset"]["split"]["cross_domain"]
            splitting = [split_config["train"], split_config["validation"], split_config["test"]]
            train_dataset, val_dataset, test_dataset = cdmetadl.dataset.split_by_names(meta_dataset, splitting)
        case DomainType.WITHIN_DOMAIN:
            splitting = [args.train_split, args.validation_split, args.test_split]
            train_dataset, val_dataset, test_dataset = cdmetadl.dataset.random_class_split(
                meta_dataset, splitting, seed=args.seed
            )

    with open(args.dataset_output_dir / 'train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(args.dataset_output_dir / 'val_dataset.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)
    with open(args.dataset_output_dir / 'test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)

    train_config = cdmetadl.config.DatasetConfig.from_json(args.config["dataset"]["train"])
    model_module = cdmetadl.helpers.general_helpers.load_module_from_path(args.model_dir / "model.py")
    match model_module.MyMetaLearner.data_format:
        case cdmetadl.config.DataFormat.TASK:
            meta_train_generator = cdmetadl.dataset.SampleTaskGenerator(train_dataset, train_config)
        case cdmetadl.config.DataFormat.BATCH:
            meta_train_generator = cdmetadl.dataset.BatchGenerator(train_dataset, train_config)

    valid_config = cdmetadl.config.DatasetConfig.from_json(args.config["dataset"]["validation"])
    meta_val_generator = cdmetadl.dataset.TaskGenerator(val_dataset, valid_config)
    return meta_train_generator, meta_val_generator


def meta_learn(
    args: argparse.Namespace, meta_train_generator: cdmetadl.dataset.DataGenerator,
    meta_val_generator: cdmetadl.dataset.TaskGenerator
) -> None:
    model_module = cdmetadl.helpers.general_helpers.load_module_from_path(args.model_dir / "model.py")

    logger = cdmetadl.logger.Logger(
        args.logger_dir, args.tensorboard_output_dir if args.use_tensorboard else None,
        meta_val_generator.dataset.number_of_datasets
    )

    model_config = cdmetadl.config.ModelConfig.from_json(args.config["model"])
    meta_learner = model_module.MyMetaLearner(
        model_config, meta_train_generator.number_of_classes, meta_train_generator.total_number_of_classes, logger
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

    with open(args.output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(args.config, f)

    cdmetadl.helpers.general_helpers.vprint("\nMeta-train your meta-learner...", args.verbose)
    meta_learn(args, meta_train_generator, meta_val_generator)
    cdmetadl.helpers.general_helpers.vprint("[+] Meta-learner meta-trained", args.verbose)


if __name__ == "__main__":
    main()
3
