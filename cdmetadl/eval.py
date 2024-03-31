import argparse
import pickle
import pathlib
import shutil
import yaml

from tqdm import tqdm
import pandas as pd
import torch

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
    parser.add_argument('--config_path', type=pathlib.Path, required=True, help='Path to config to use.')
    parser.add_argument(
        '--training_output_dir', type=pathlib.Path, required=True, help='Path to the output directory for training'
    )
    parser.add_argument(
        '--data_dir', type=pathlib.Path, default=None,
        help='Default location of the directory containing the meta_train and meta_test data. Default: None.'
    )
    parser.add_argument(
        '--output_dir', type=pathlib.Path, default='./output/tmp/eval',
        help='Path to the output directory for evaluation. Default: "./output/tmp/eval".'
    )
    parser.add_argument(
        '--overwrite_previous_results', action=argparse.BooleanOptionalAction, default=False,
        help='Overwrites the previous output directory instead of giving an error. Default: False.'
    )
    return parser


def process_args(args: argparse.Namespace) -> None:
    args.training_output_dir = args.training_output_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.resolve()

    args.config = cdmetadl.config.read_eval_config(args.config_path)

    with open(args.training_output_dir / "config.yaml", "r") as f:
        args.training_config = yaml.safe_load(f)

    args.model_dir = pathlib.Path(args.training_config["model"]["path"]).resolve()
    args.device = cdmetadl.helpers.general_helpers.get_device()


def prepare_directories(args: argparse.Namespace) -> None:
    cdmetadl.helpers.general_helpers.exist_dir(args.training_output_dir)
    cdmetadl.helpers.general_helpers.exist_dir(args.model_dir)

    training_structure = args.training_output_dir.parts[args.training_output_dir.parts.index("training") + 1:]

    args.output_dir /= f"train_cfg_{training_structure[0]}"  # Training config name
    args.output_dir /= f"eval_cfg_{args.config_path.stem}"  # Evaluation config name
    args.output_dir /= pathlib.Path(
        *training_structure[1:]
    )  # rest of folder structure like model name, evaluation mode, dataset, ...
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

    # Hacky fix to make datasets portable
    if args.data_dir is not None:
        for dataset in test_dataset.datasets:
            dataset.img_paths = [args.data_dir / path.relative_to(path.parents[-4]) for path in dataset.img_paths]

    test_generator_config = cdmetadl.config.DatasetConfig.from_json(args.config["dataset"]["test"])
    return cdmetadl.dataset.TaskGenerator(test_dataset, test_generator_config)


def initalize_confidence_estimator(args: argparse.Namespace) -> cdmetadl.confidence.ConfidenceEstimator:
    confidence_estimator_config = args.config["evaluation"]["confidence_estimators"]
    match confidence_estimator_config["use"]:
        case "ConstantConfidenceProvider":
            return cdmetadl.confidence.ConstantConfidenceProvider(
                **confidence_estimator_config["ConstantConfidenceProvider"]
            )
        case "MCDropoutConfidenceEstimator":
            return cdmetadl.confidence.MCDropoutConfidenceEstimator(
                **confidence_estimator_config["MCDropoutConfidenceEstimator"]
            )
        case "PseudoConfidenceEstimator":
            return cdmetadl.confidence.PseudoConfidenceEstimator(
                **confidence_estimator_config["PseudoConfidenceEstimator"]
            )
        case "GTConfidenceEstimator":
            return cdmetadl.confidence.GTConfidenceEstimator(**confidence_estimator_config["GTConfidenceEstimator"])
        case _:
            raise ValueError(
                f'Confidence estimator {args.config["evaluation"]["confidence_estimators"]} specified in config does not exists.'
            )


def initalize_data_augmentor(args: argparse.Namespace) -> cdmetadl.augmentation.Augmentation:
    data_augmentor_config = args.config["evaluation"]["augmentors"]
    match data_augmentor_config["use"]:
        case None:
            return None
        case "PseudoAugmentation":
            return cdmetadl.augmentation.PseudoAugmentation(
                **data_augmentor_config["PseudoAugmentation"], device=args.device
            )
        case "StandardAugmentation":
            return cdmetadl.augmentation.StandardAugmentation(
                **data_augmentor_config["StandardAugmentation"], device=args.device
            )
        case "GenerativeAugmentation":
            return cdmetadl.augmentation.GenerativeAugmentation(
                **data_augmentor_config["GenerativeAugmentation"], device=args.device
            )
        case _:
            raise ValueError(
                f'Data augmentor {args.config["evaluation"]["augmentors"]} specified in config does not exists.'
            )


def meta_test(args: argparse.Namespace, meta_test_generator: cdmetadl.dataset.TaskGenerator) -> list[dict]:
    model_module = cdmetadl.helpers.general_helpers.load_module_from_path(args.model_dir / "model.py")
    confidence_learner: cdmetadl.api.Learner = model_module.MyLearner()
    learner: cdmetadl.api.Learner = model_module.MyLearner()

    confidence_estimator = initalize_confidence_estimator(args)
    augmentor = initalize_data_augmentor(args)
    tasks_per_dataset = args.config["evaluation"]["tasks_per_dataset"]

    predictions = []
    total_number_of_tasks = meta_test_generator.dataset.number_of_datasets * tasks_per_dataset
    for task in tqdm(meta_test_generator(tasks_per_dataset), total=total_number_of_tasks):
        task.support_set.images = task.support_set.images.to(args.device)
        task.support_set.labels = task.support_set.labels.to(args.device)
        task.query_set.images = task.query_set.images.to(args.device)
        task.query_set.labels = task.query_set.labels.to(args.device)

        original_number_of_shots = task.support_set.min_number_of_shots

        if type(confidence_estimator) is cdmetadl.confidence.GTConfidenceEstimator:
            confidence_estimator.set_query_set(task.query_set)

        confidence_learner.load(args.training_output_dir / "model")
        # Adjust T for finetuning
        if "finetuning" in str(model_module).lower():
            T = 1000
        elif "maml" in str(model_module).lower():
            T = 35
            
        confidence_learner.T = T
        task.support_set, confidence_scores = confidence_estimator.estimate(confidence_learner, task.support_set)

        if augmentor is not None:
            # print(confidence_scores)
            task.support_set = augmentor.augment(task.support_set, conf_scores=confidence_scores)

        learner.load(args.training_output_dir / "model")
        # Adjust T for finetuning
        learner.T = T
        predictor = learner.fit(task.support_set)

        predictions.append({
            "Dataset": task.dataset_name,
            "Number of Ways": task.number_of_ways,
            "Number of Shots": original_number_of_shots,
            "Number of Shots per Class": task.support_set.number_of_shots_per_class,
            "Confidence Scores": confidence_scores,
            "Predictions": predictor.predict(task.query_set.images).cpu().numpy(),
            "Ground Truth": task.query_set.labels.cpu().numpy(),
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
    with open(args.output_dir / "training_config.yaml", "w") as f:
        yaml.safe_dump(args.training_config, f)

    cdmetadl.helpers.general_helpers.vprint("\nMeta-testing your learner...", args.verbose)
    predictions = meta_test(args, meta_test_generator)
    cdmetadl.helpers.general_helpers.vprint("[+] Learner meta-tested", args.verbose)

    cdmetadl.helpers.general_helpers.vprint("\nEvaluate learner...", args.verbose)
    evaluate(args, predictions)
    cdmetadl.helpers.general_helpers.vprint("[+] Evaluated", args.verbose)


if __name__ == "__main__":
    main()
