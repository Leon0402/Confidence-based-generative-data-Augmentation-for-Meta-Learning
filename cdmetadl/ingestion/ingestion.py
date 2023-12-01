""" This is the ingestion program written by the organizers. This program also
runs on the Cross-Domain MetaDL competition platform to test your code.

Usage: 

python -m cdmetadl.ingestion.ingestion \
    --input_data_dir=input_dir \
    --output_dir_ingestion=output_dir \
    --submission_dir=submission_dir

* The input directory input_dir (e.g. ../../public_data) contains several 
datasets formatted following this instructions: 
https://github.com/ihsaan-ullah/meta-album/tree/master/DataFormat

* The output directory output_dir (e.g. ../../ingestion_output) will store 
the predicted labels of the meta-testing phase, the metadata, the meta-trained 
learner, and the logs:
    logs/                   <- Directory containing all the meta-training logs
    model/                  <- Meta-trained learner (Output of Learner.save())
    metadata_ingestion      <- Metadata from the ingestion program
	task_{task_id}.predict  <- Predicter probabilities for each meta-test task

* The code directory submission_dir (e.g. ../../baselines/random) must 
contain your code submission model.py, it can also contain other helpers and
configuration files.

AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
"""
import os
import time
import datetime
import sys
import shutil
from pathlib import Path
import pickle

import numpy as np

import cdmetadl.dataset
from cdmetadl.helpers.ingestion_helpers import *
from cdmetadl.helpers.general_helpers import *
from cdmetadl.ingestion.competition_logger import Logger

# Program version
VERSION = 1.1


def read_generator_configs(config_file: Path) -> Tuple[str, dict, dict, dict]:
    train_data_format = "task"

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
        user_config = load_json(config_file)
        if "train_data_format" in user_config:
            train_data_format = user_config["train_data_format"]
        if "train_config" in user_config:
            train_generator_config.update(user_config["train_config"])
        if "valid_config" in user_config:
            valid_generator_config.update(user_config["valid_config"])

    return train_data_format, train_generator_config, valid_generator_config, test_generator_config


def ingestion(args) -> None:
    total_time_start = time.time()

    vprint(f"Ingestion program version: {VERSION}", args.verbose)
    vprint(f"Using random seed: {args.seed}", args.verbose)

    # Define the path to the directories
    input_dir = args.input_data_dir.resolve()
    output_dir = args.output_dir_ingestion.resolve()
    submission_dir = args.submission_dir.resolve()

    # Show python version and directory structure
    print(f"\nPython version: {sys.version}")
    print("\n\n")
    os.system("nvidia-smi")
    print("\n\n")
    os.system("nvcc --version")
    print("\n\n")
    os.system("pip list")
    print("\n\n")

    if args.debug_mode >= 2:
        print(f"Using input_dir: {input_dir}")
        print(f"Using output_dir: {output_dir}")
        print(f"Using submission_dir: {submission_dir}")
        show_dir(".")

    if args.debug_mode == 3:
        join_list = lambda info: "\n".join(info)
        gpu_settings = ["\n----- GPU settings -----"]
        gpu_info = get_torch_gpu_environment()
        gpu_settings.extend(gpu_info)
        print(join_list(gpu_settings))
        sys.exit(0)

    # Import your model
    try:
        sys.path.append(str(submission_dir))
        from model import MyMetaLearner, MyLearner
    except:
        print(f"MyMetaLearner and MyLearner not found in {submission_dir}/model.py")
        sys.exit(1)

    vprint(f"\n{'#'*60}\n{'#'*17} Ingestion program starts {'#'*17}\n{'#'*60}", args.verbose)

    # Check all the required directories
    vprint("\nChecking directories...", args.verbose)
    exist_dir(input_dir)
    exist_dir(submission_dir)

    # Create output dir
    if output_dir.exists() and not args.overwrite_previous_results:
        timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        output_dir.rename(output_dir.parent / f"{output_dir.name}_{timestamp}")
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Create logs dir and initialize logger
    logs_dir = output_dir / "logs"
    logs_dir.mkdir()
    logger = Logger(logs_dir)

    # Create output model dir
    model_dir = output_dir / "model"
    model_dir.mkdir()

    vprint("[+] Directories", args.verbose)

    vprint("\nDefining generators config..", args.verbose)
    train_data_format, train_generator_config, valid_generator_config, test_generator_config = read_generator_configs(
        submission_dir / "config.json"
    )

    vprint("\nPreparing dataset", args.verbose)

    datasets_info = check_datasets(input_dir, args.datasets, args.verbose)
    dataset = cdmetadl.dataset.MetaImageDataset([
        cdmetadl.dataset.ImageDataset(name, info, args.image_size) for name, info in datasets_info.items()
    ])

    if args.domain_type == 'cross-domain':
        train_dataset, val_dataset, test_dataset = cdmetadl.dataset.random_meta_split(dataset, args.split_sizes)
    elif args.domain_type == 'within-domain':
        train_dataset, val_dataset, test_dataset = cdmetadl.dataset.random_class_split(dataset, args.split_sizes)

    with open(output_dir / 'test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)

    total_classes = sum(dataset.number_of_classes for dataset in train_dataset.datasets)
    if train_data_format == "task":
        meta_train_generator = cdmetadl.dataset.create_task_generator(train_dataset, train_generator_config)
        # TODO(leon): Isn't "train_classes" dynamic here in any-way any-shot?
        train_classes = train_generator_config["N"]
    else:
        meta_train_generator = cdmetadl.dataset.create_batch_generator(train_dataset)
        train_classes = total_classes

    meta_val_generator = cdmetadl.dataset.create_task_generator(
        val_dataset, valid_generator_config, sample_dataset=True
    )
    meta_test_generator = cdmetadl.dataset.create_task_generator(test_dataset, test_generator_config)

    # Save/print experimental settings
    join_list = lambda info: "\n".join(info)

    gpu_settings = ["\n----- GPU settings -----"]
    gpu_info = get_torch_gpu_environment()
    gpu_settings.extend(gpu_info)

    # data_settings = [
    #     "\n----- Data settings -----", f"# Train datasets: {len(train_datasets_info)}",
    #     f"# Validation datasets: {len(valid_datasets_info)}", f"# Test datasets: {len(test_datasets_info)}",
    #     f"Image size: {args.image_size}", f"Random seed: {args.seed}"
    # ]

    # if train_data_format == "task":
    #     train_settings = [
    #         "\n----- Train settings -----", f"Train data format: {train_data_format}", f"N-way: {train_loader.n_way}",
    #         f"Minimum ways: {train_loader.min_ways}", f"Maximum ways: {train_loader.max_ways}",
    #         f"k-shot: {train_loader.k_shot}", f"Minimum shots: {train_loader.min_shots}",
    #         f"Maximum shots: {train_loader.max_shots}", f"Query size: {train_loader.query_size}"
    #     ]
    # else:
    #     train_settings = [
    #         "\n----- Train settings -----", f"Train data format: {train_data_format}", f"Batch size: {batch_size}"
    #     ]

    # if len(valid_datasets_info) > 0:
    #     validation_settings = [
    #         "\n----- Validation settings -----", f"N-way: {valid_loader.n_way}",
    #         f"Minimum ways: {valid_loader.min_ways}", f"Maximum ways: {valid_loader.max_ways}",
    #         f"k-shot: {valid_loader.k_shot}", f"Minimum shots: {valid_loader.min_shots}",
    #         f"Maximum shots: {valid_loader.max_shots}", f"Query size: {valid_loader.query_size}"
    #     ]

    # test_settings = [
    #     "\n----- Test settings -----", f"N-way: {test_loader.n_way}", f"Minimum ways: {test_loader.min_ways}",
    #     f"Maximum ways: {test_loader.max_ways}", f"k-shot: {test_loader.k_shot}",
    #     f"Minimum shots: {test_loader.min_shots}", f"Maximum shots: {test_loader.max_shots}",
    #     f"Query size: {test_loader.query_size}"
    # ]

    # if len(valid_datasets_info) > 0:
    #     all_settings = [
    #         f"\n{'*'*9} Experimental settings {'*'*9}",
    #         join_list(gpu_settings),
    #         join_list(data_settings),
    #         join_list(train_settings),
    #         join_list(validation_settings),
    #         join_list(test_settings), f"\n{'*'*41}"
    #     ]
    # else:
    #     all_settings = [
    #         f"\n{'*'*9} Experimental settings {'*'*9}",
    #         join_list(gpu_settings),
    #         join_list(data_settings),
    #         join_list(train_settings),
    #         join_list(test_settings), f"\n{'*'*41}"
    #     ]

    # experimental_settings = join_list(all_settings)
    # vprint(experimental_settings, args.verbose)
    # experimental_settings_file = f"{logs_dir}/experimental_settings.txt"
    # with open(experimental_settings_file, "w") as f:
    #     f.writelines(experimental_settings)

    # Meta-train
    vprint("\nMeta-training your meta-learner...", args.verbose)
    meta_training_start = time.time()
    meta_learner = MyMetaLearner(train_classes, total_classes, logger)
    learner = meta_learner.meta_fit(meta_train_generator, meta_val_generator)
    meta_training_time = time.time() - meta_training_start
    learner.save(model_dir)
    vprint("[+] Meta-learner meta-trained", args.verbose)

    # Meta-test
    vprint("\nMeta-testing your learner...", args.verbose)
    meta_testing_start = time.time()
    for i, task in enumerate(meta_test_generator(args.test_tasks_per_dataset)):
        vprint(f"\tTask {i+1} started...", args.verbose)
        learner = MyLearner()
        learner.load(model_dir)

        support_set = (task.support_set[0], task.support_set[1], task.support_set[2], task.num_ways, task.num_shots)

        task_start = time.time()
        predictor = learner.fit(support_set)
        vprint("\t\t[+] Learner trained", args.verbose)

        y_pred = predictor.predict(task.query_set[0])
        vprint("\t\t[+] Labels predicted", args.verbose)

        task_time = time.time() - task_start
        if args.debug_mode >= 1:
            if task_time > args.max_time:
                print(
                    f"\t\t[-] Task {i+1} exceeded the maximum time allowed. " +
                    f"Max time {args.max_time}, execution time {task_time}"
                )
                exit(1)

        fmt = "%f" if len(y_pred.shape) == 2 else "%d"
        np.savetxt(output_dir / f"task_{i+1}.predict", y_pred, fmt=fmt)
        with open(output_dir / f"task_{i+1}.pkl", 'wb') as f:
            pickle.dump(task, f)

        vprint("\t\t[+] Predictions saved", args.verbose)

        vprint(f"\t[+] Task {i+1} finished in {task_time} seconds", args.verbose)
    vprint("[+] Learner meta-tested", args.verbose)
    meta_testing_time = time.time() - meta_testing_start

    total_time = time.time() - total_time_start
    with open(f"{output_dir}/metadata_ingestion", "w") as global_metadata_file:
        global_metadata_file.write(f"Total execution time: {total_time}\n")
        global_metadata_file.write(f"Meta-train time: {meta_training_time}\n")
        global_metadata_file.write(f"Meta-test time: {meta_testing_time}\n")
        global_metadata_file.write(f"Number of test datasets: {test_dataset.number_of_datasets}\n")
        global_metadata_file.write(f"Tasks per dataset: {args.test_tasks_per_dataset}")
    vprint(f"\nOverall time spent: {total_time} seconds", args.verbose)

    vprint(f"\n{'#'*60}\n{'#'*9} Ingestion program finished successfully " + f"{'#'*10}\n{'#'*60}", args.verbose)
