import argparse
import pickle
import pathlib

from tqdm import tqdm
import torch
import pandas as pd

import cdmetadl.dataset
import cdmetadl.helpers.general_helpers
import cdmetadl.helpers.scoring_helpers
import cdmetadl.dataset.split
import cdmetadl.confidence_estimator
import cdmetadl.augmentation


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
    parser.add_argument(
        '--pseudo_DA', action=argparse.BooleanOptionalAction, default=False,
        help='Uses pseudo data augmentation. Default: False.'
    )

    return parser


def process_args(args: argparse.Namespace) -> None:
    args.training_output_dir = args.training_output_dir.resolve()
    args.model_dir = args.model_dir.resolve()
    args.output_dir = args.output_dir.resolve()


def prepare_directories(args: argparse.Namespace) -> None:
    cdmetadl.helpers.general_helpers.exist_dir(args.training_output_dir)
    cdmetadl.helpers.general_helpers.exist_dir(args.model_dir)

    # TODO: How to deal with existing output dir
    args.output_dir /= args.training_output_dir.relative_to(args.training_output_dir.parent.parent.parent)
    args.output_dir.mkdir(parents=True, exist_ok=True)


def prepare_data_generators(args: argparse.Namespace) -> cdmetadl.dataset.TaskGenerator:
    with open(args.training_output_dir / "datasets/test_dataset.pkl", 'rb') as f:
        test_dataset: cdmetadl.dataset.MetaImageDataset = pickle.load(f)

    test_generator_config = {
        "N": None,
        "min_N": 2,
        "max_N": 5,
        "k": None,
        "min_k": 1,
        "max_k": 20,
        "query_images_per_class": 20
    }

    return cdmetadl.dataset.TaskGenerator(test_dataset, test_generator_config)   




def meta_test(args: argparse.Namespace, meta_test_generator: cdmetadl.dataset.TaskGenerator, augmentation: False) -> list[dict]:
    model_module = cdmetadl.helpers.general_helpers.load_module_from_path(args.model_dir / "model.py")

    predictions = []
    total_number_of_tasks = meta_test_generator.dataset.number_of_datasets * args.test_tasks_per_dataset
    for task in tqdm(meta_test_generator(args.test_tasks_per_dataset), total=total_number_of_tasks):
        # get confidence score (call helper perform prediction step)
        # sample more images from DS here (call augmentation.py)
        print("getting confidence estimation")
        support_set = (task.support_set[0], task.support_set[1], task.support_set[2], task.num_ways, task.num_shots)

        learner = model_module.MyLearner()
        learner.load(args.training_output_dir / "model")
        
        print("splitting sets:")
        support_set, conf_support_set, backup_support_set, query_set, conf_query_set = rand_conf_split(task.support_set, task.query_set, task.num_ways, task.num_shots)
        
        conf_scores = ref_set_confidence_scores(conf_support_set, conf_query_set, learner)
        print("conf_scores: ", conf_scores)

        # augment step here
        augmentation = PseudoAug(task.support_set, conf_support_set, conf_scores, threshold, scale)
        support_set = augmentation.getDatasetAugmented()


        # rewrite this
        predictor = learner.fit(support_set)

        predictions.append({
            "Dataset": task.dataset,
            "Number of Ways": task.num_ways,
            "Number of Shots": task.num_shots,
            "Predictions": predictor.predict(task.query_set[0]),
            "Ground Truth": task.query_set[1].numpy(),
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

    # check for data augmentation, confidence estimation step, augment tasks (e.g with test set)
    if args.pseudo_DA: 
        print("using confidence estimation, DA")



    cdmetadl.helpers.general_helpers.vprint("\nMeta-testing your learner...", args.verbose)
    predictions = meta_test(args, meta_test_generator, args.pseudo_DA)
    cdmetadl.helpers.general_helpers.vprint("[+] Learner meta-tested", args.verbose)

    cdmetadl.helpers.general_helpers.vprint("\nEvaluate learner...", args.verbose)
    evaluate(args, predictions)
    cdmetadl.helpers.general_helpers.vprint("[+] Evaluated", args.verbose)


if __name__ == "__main__":
    main()
