import argparse
import pickle
import pathlib
import shutil
import yaml

from tqdm import tqdm
import pandas as pd

import cdmetadl.api
import cdmetadl.augmentation
import cdmetadl.confidence
import cdmetadl.config
import cdmetadl.dataset
import cdmetadl.samplers
import cdmetadl.helpers.general_helpers
import cdmetadl.helpers.scoring_helpers


def define_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument(
        '--verbose', action=argparse.BooleanOptionalAction, default=True,
        help='Show various progression messages. Default: True.'
    )
    parser.add_argument('--seed', type=int, default=42, help='Seed to make results reproducible. Default: 42.')
    parser.add_argument(
        '--training_output_dir', type=pathlib.Path, required=True, help='Path to the output directory for training'
    )
    parser.add_argument(
        '--output_dir', type=pathlib.Path, default='./output/tmp/eval',
        help='Path to the output directory for evaluation. Default: "./output/tmp/eval".'
    )
    parser.add_argument(
        '--test_tasks_per_dataset', type=int, default=100,
        help='The total number of test tasks will be num_datasets x test_tasks_per_dataset. Default: 100.'
    )
    parser.add_argument(
        '--pseudo_DA', action=argparse.BooleanOptionalAction, default=False,
        help='Uses pseudo data augmentation. Default: False.'
    )
    parser.add_argument(
        '--overwrite_previous_results', action=argparse.BooleanOptionalAction, default=False,
        help='Overwrites the previous output directory instead of renaming it. Default: False.'
    )
    return parser


def process_args(args: argparse.Namespace) -> None:
    args.training_output_dir = args.training_output_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    with open(args.training_output_dir / "config.yaml", "r") as f:
        args.config = yaml.safe_load(f)
    args.model_dir = pathlib.Path(args.config["model"]["path"]).resolve()
    args.config["model"]["number_of_test_tasks_per_dataset"] = args.test_tasks_per_dataset


def prepare_directories(args: argparse.Namespace) -> None:
    cdmetadl.helpers.general_helpers.exist_dir(args.training_output_dir)
    cdmetadl.helpers.general_helpers.exist_dir(args.model_dir)

    args.output_dir /= pathlib.Path(
        *args.training_output_dir.parts[args.training_output_dir.parts.index("training") + 1:]
    )
    if args.output_dir.exists() and args.overwrite_previous_results:
        shutil.rmtree(args.output_dir)
    elif args.output_dir.exists():
        raise ValueError(
            f"Output directory {args.output_dir} exists and overwrite previous results option is not provided"
        )
    args.output_dir.mkdir(parents=True)


def prepare_data_generators(args: argparse.Namespace) -> cdmetadl.dataset.TaskGenerator:
    with open(args.training_output_dir / "datasets/test_dataset.pkl", 'rb') as f:
        test_dataset: cdmetadl.dataset.MetaImageDataset = pickle.load(f)

    test_generator_config = cdmetadl.config.DatasetConfig.from_json(args.config["dataset"]["test"])
    return cdmetadl.dataset.TaskGenerator(test_dataset, test_generator_config)


def meta_test(args: argparse.Namespace, meta_test_generator: cdmetadl.dataset.TaskGenerator) -> list[dict]:
    model_module = cdmetadl.helpers.general_helpers.load_module_from_path(args.model_dir / "model.py")

    confidence_estimator = cdmetadl.confidence.ConstantConfidenceProvider(confidence=1)
    confidence_estimator = cdmetadl.confidence.PseudoConfidenceEstimator()
    confidence_estimator = cdmetadl.confidence.MCDropoutConfidenceEstimator(num_samples=20)

    augmentor: cdmetadl.augmentation.Augmentation = None
    augmentor = cdmetadl.augmentation.PseudoAugmentation(threshold=0.75, scale=2, keep_original_data=True)
    augmentor = cdmetadl.augmentation.StandardAugmentation(threshold=0.75, scale=2, keep_original_data=True)
    augmentor = cdmetadl.augmentation.GenerativeAugmentation(threshold=0.75, scale=2, keep_original_data=True)

    predictions = []
    total_number_of_tasks = meta_test_generator.dataset.number_of_datasets * args.test_tasks_per_dataset
    for task in tqdm(meta_test_generator(args.test_tasks_per_dataset), total=total_number_of_tasks):

        learner: cdmetadl.api.Learner = model_module.MyLearner()
        learner.load(args.training_output_dir / "model")

        task.support_set, confidence_scores = confidence_estimator.estimate(learner, task.support_set)

        if augmentor is not None:
            task.support_set = augmentor.augment(task.support_set, conf_scores=confidence_scores)

        predictor = learner.fit(task.support_set)

        predictions.append({
            "Dataset": task.dataset_name,
            "Number of Ways": task.number_of_ways,
            # TODO: Save all number of shots rather than min?
            "Number of Shots": task.support_set.min_number_of_shots,
            "Predictions": predictor.predict(task.query_set.images),
            "Ground Truth": task.query_set.labels.numpy(),
        })

    with open(args.output_dir / f"predictions.pkl", 'wb') as f:
        pickle.dump(predictions, f)

    return predictions


def evaluate(args: argparse.Namespace, predictions: list[dict]):
    evaluations = [{
        "Dataset": pred_dict["Dataset"],
        "Number of Ways": pred_dict["Number of Ways"],
        "Number of Shots": pred_dict["Number of Shots"],
        **cdmetadl.helpers.scoring_helpers.compute_all_scores(
            pred_dict["Ground Truth"], pred_dict["Predictions"], pred_dict["Number of Ways"]
        ),
    } for pred_dict in tqdm(predictions)]
    pd.DataFrame(evaluations).to_pickle(args.output_dir / "evaluation.pkl")


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
    meta_test_generator = prepare_data_generators(args)
    cdmetadl.helpers.general_helpers.vprint("[+]Datasets prepared", args.verbose)

    with open(args.output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(args.config, f)

    cdmetadl.helpers.general_helpers.vprint("\nMeta-testing your learner...", args.verbose)
    predictions = meta_test(args, meta_test_generator)
    cdmetadl.helpers.general_helpers.vprint("[+] Learner meta-tested", args.verbose)

    cdmetadl.helpers.general_helpers.vprint("\nEvaluate learner...", args.verbose)
    evaluate(args, predictions)
    cdmetadl.helpers.general_helpers.vprint("[+] Evaluated", args.verbose)


if __name__ == "__main__":
    main()
