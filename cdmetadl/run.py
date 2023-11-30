""" Combine the ingestion and scoring processes. 

Usage: 
    python -m cdmetadl.run \
        --input_data_dir=../public_data \
        --submission_dir=../baselines/random \
        --overwrite_previous_results=True \
        --test_tasks_per_dataset=10
    
AS A PARTICIPANT, DO NOT MODIFY THIS CODE. 
"""
import argparse
from pathlib import Path

from cdmetadl.ingestion import ingestion as cdmetadl_ingestion
from cdmetadl.scoring import scoring as cdmetadl_scoring


def main():
    """Runs the ingestion and scoring programs sequentially"""

    parser = argparse.ArgumentParser(description='Scoring')
    parser.add_argument(
        '--seed', type=int, default=93, help='Any int to be used as random seed for reproducibility. Default: 93.'
    )
    parser.add_argument(
        '--verbose',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help=
        'True: show various progression messages (recommended); False: no progression messages are shown. Default: True.'
    )
    parser.add_argument(
        '--debug_mode',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help=
        '0: no debug; 1: compute additional scores (recommended); 2: same as 1 + show the Python version and list the directories. Default: 1.'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=128,
        help='Int specifying the image size for all generators (recommended value 128). Default: 128.'
    )
    parser.add_argument(
        '--private_information',
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help=
        'True: the name of the datasets is kept private; False: all information is shown (recommended). Default: False.'
    )
    parser.add_argument(
        '--overwrite_previous_results',
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help=
        'True: the previous output directory is overwritten; False: the previous output directory is renamed (recommended). Default: False.'
    )
    parser.add_argument(
        '--max_time', type=int, default=1000, help='Maximum time in seconds PER TESTING TASK. Default: 1000.'
    )
    parser.add_argument(
        '--test_tasks_per_dataset',
        type=int,
        default=100,
        help='The total number of test tasks will be num_datasets x test_tasks_per_dataset. Default: 100.'
    )
    parser.add_argument(
        '--domain_type',
        choices=['cross-domain', 'within-domain'],
        default="cross-domain",
        help=
        "Choose the domain type for meta-learning. 'cross-domain' indicates a setup where multiple distinct datasets are utilized, ensuring that the training, validation, and testing sets are entirely separate, thereby promoting generalization across diverse data sources. 'within-domain', on the other hand, refers to using a single dataset, segmented into different classes for training, validation, and testing, focusing on learning variations within a more homogeneous data environment. Default: cross-domain."
    )
    parser.add_argument(
        '--input_data_dir',
        type=Path,
        default='../../public_data',
        help=
        'Default location of the directory containing the meta_train and meta_test data. Default: "../../public_data".'
    )
    parser.add_argument(
        '--results_dir',
        type=Path,
        default='../../ingestion_output',
        help='Default location of the output directory for the ingestion program. Default: "../../ingestion_output".'
    )
    parser.add_argument(
        '--output_dir_scoring',
        type=Path,
        default='../../scoring_output',
        help='Default location of the output directory for the scoring program. Default: "../../scoring_output".'
    )
    parser.add_argument(
        '--output_dir_ingestion',
        type=Path,
        default='../../ingestion_output',
        help='Path to the output directory for the ingestion program. Default: "../../ingestion_output".'
    )
    parser.add_argument(
        '--submission_dir',
        type=Path,
        default='../../baselines/random',
        help='Path to the directory containing the solution to use. Default: "../../baselines/random".'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default= "BCT, BRD, CRS, FLW, MD_MIX, PLK",
        help='Specify the datasets that will be used for. Default: "BCT, BRD, CRS, FLW, MD_MIX, PLK"'
    )
    parser.add_argument(
        '--split_size',
        type=str,
        default= "0.3, 0.3, 0.4",
        help='''Defines how much of the data will be assigned to the training, testing and validation data.
                Please ensure that the values sum up to one.
                In the order: training, validation, test.
                Default: "0.3, 0.3, 0.4"'''
    )


    args = parser.parse_args()
    cdmetadl_ingestion.ingestion(args)

    args.results_dir = args.output_dir_ingestion
    cdmetadl_scoring.scoring(args)


if __name__ == "__main__":
    main()
