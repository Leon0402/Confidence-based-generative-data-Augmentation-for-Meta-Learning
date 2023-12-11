import argparse
import pickle
import pathlib

import tqdm.contrib
import numpy as np
import pandas as pd

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers
import cdmetadl.helpers.scoring_helpers


def define_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument(
        '--verbose', action=argparse.BooleanOptionalAction, default=True,
        help='Show various progression messages. Default: True.'
    )
    parser.add_argument(
        '--training_output_dir', type=pathlib.Path, default="./training_output",
        help='Path to the output directory for training. Default: "./training_output".'
    )
    parser.add_argument(
        '--model_dir', type=pathlib.Path, required=True, help='Path to the directory containing the solution to use.'
    )
    parser.add_argument(
        '--output_dir', type=pathlib.Path, default='./eval_output',
        help='Path to the output directory for evaluation. Default: "./eval_output".'
    )
    parser.add_argument(
        '--test_tasks_per_dataset', type=int, default=100,
        help='The total number of test tasks will be num_datasets x test_tasks_per_dataset. Default: 100.'
    )

    return parser


def process_args(args: argparse.Namespace) -> None:
    args.training_output_dir = args.training_output_dir.resolve()
    args.model_dir = args.model_dir.resolve()
    args.output_dir = args.output_dir.resolve()


def prepare_directories(args: argparse.Namespace) -> None:
    args.training_output_dir /= args.model_dir.name
    cdmetadl.helpers.general_helpers.exist_dir(args.training_output_dir)
    cdmetadl.helpers.general_helpers.exist_dir(args.model_dir)

    # TODO: How to deal with existing output dir
    args.output_dir /= args.model_dir.name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.prediction_output_dir = args.output_dir / "predictions"
    args.prediction_output_dir.mkdir(exist_ok=True)
    args.task_output_dir = args.output_dir / "tasks"
    args.task_output_dir.mkdir(exist_ok=True)
    args.eval_output_dir = args.output_dir / "eval"
    args.eval_output_dir.mkdir(exist_ok=True)


def prepare_data_generators(args: argparse.Namespace) -> cdmetadl.dataset.TaskGenerator:
    with open(args.training_output_dir / "datasets/test_dataset.pkl", 'rb') as f:
        test_dataset: cdmetadl.dataset.MetaImageDataset = pickle.load(f)

    test_generator_config = {
        "N": None,
        "min_N": 2,
        "max_N": 20,
        "k": None,
        "min_k": 1,
        "max_k": 20,
        "query_images_per_class": 20
    }

    return cdmetadl.dataset.TaskGenerator(test_dataset, test_generator_config)


def meta_test(args: argparse.Namespace, meta_test_generator: cdmetadl.dataset.TaskGenerator) -> None:
    model_module = cdmetadl.helpers.general_helpers.load_module_from_path(args.model_dir / "model.py")

    total_number_of_tasks = meta_test_generator.dataset.number_of_datasets * args.test_tasks_per_dataset
    for i, task in tqdm.contrib.tenumerate(
        meta_test_generator(args.test_tasks_per_dataset), total=total_number_of_tasks
    ):
        support_set = (task.support_set[0], task.support_set[1], task.support_set[2], task.num_ways, task.num_shots)

        learner = model_module.MyLearner()
        learner.load(args.training_output_dir / "model")
        predictor = learner.fit(support_set)

        y_pred = predictor.predict(task.query_set[0])

        np.savetxt(
            args.prediction_output_dir / f"task_{i+1}.predict", y_pred, fmt="%f" if len(y_pred.shape) == 2 else "%d"
        )
        with open(args.task_output_dir / f"task_{i+1}.pkl", 'wb') as f:
            pickle.dump(task, f)


def evaluate(args: argparse.Namespace):
    data = []
    for task_file, pred_file in zip(
        sorted(args.task_output_dir.iterdir()), sorted(args.prediction_output_dir.iterdir())
    ):
        with open(task_file, 'rb') as f:
            task: cdmetadl.dataset.Task = pickle.load(f)

        y_pred = cdmetadl.helpers.scoring_helpers.read_results_file(pred_file)

        task_scores = cdmetadl.helpers.scoring_helpers.compute_all_scores(
            task.query_set[1].numpy(), y_pred, task.num_ways
        )
        task_scores['Dataset'] = task.dataset
        task_scores['Number of Ways'] = task.num_ways
        task_scores['Number of Shots'] = task.num_shots

        data.append(task_scores)

    pd.DataFrame(data).to_pickle(args.eval_output_dir / "eval.pkl")


def main():
    parser = define_argparser()
    args = parser.parse_args()

    cdmetadl.helpers.general_helpers.set_seed(42)

    cdmetadl.helpers.general_helpers.vprint("\nProcess command line arguments", args.verbose)
    process_args(args)
    cdmetadl.helpers.general_helpers.vprint("[+] Command line arguments processed", args.verbose)

    cdmetadl.helpers.general_helpers.vprint("\nPrepare directories", args.verbose)
    prepare_directories(args)
    cdmetadl.helpers.general_helpers.vprint("[+] Prepared directories", args.verbose)

    cdmetadl.helpers.general_helpers.vprint("\nPreparing dataset", args.verbose)
    meta_test_generator = prepare_data_generators(args)
    cdmetadl.helpers.general_helpers.vprint("[+]Datasets prepared", args.verbose)

    cdmetadl.helpers.general_helpers.vprint("\nMeta-testing your learner...", args.verbose)
    meta_test(args, meta_test_generator)
    cdmetadl.helpers.general_helpers.vprint("[+] Learner meta-tested", args.verbose)

    cdmetadl.helpers.general_helpers.vprint("\nEvaluate learner...", args.verbose)
    evaluate(args)
    cdmetadl.helpers.general_helpers.vprint("[+] Evaluated", args.verbose)


if __name__ == "__main__":
    main()
