# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

"""DeepFM Network trainer."""

import argparse
import json
import logging
import os
import time


from trainer.model import create_model

SNAPSHOT_PREFIX = 'deepfm'

def main(args):
    """Runs a training loop."""

    repeat_dataset = False if args.num_epochs > 0 else True

    model = create_model(
        train_data=[args.train_data],
        valid_data=args.valid_data,
        dropout_rate=args.dropout_rate,
        num_dense_features=args.num_dense_features,
        num_sparse_features=args.num_sparse_features,
        num_workers=args.num_workers,
        slot_size_array=args.slot_size_array,
        batchsize=args.batchsize,
        lr=args.lr,
        gpus=args.gpus,
        repeat_dataset=repeat_dataset)

    model.summary()

    model.fit(
        num_epochs=args.num_epochs,
        max_iter=args.max_iter,
        display=args.display_interval, 
        eval_interval=args.eval_interval, 
        snapshot=args.snapshot, 
        snapshot_prefix=SNAPSHOT_PREFIX)


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--train_data',
                        type=str,
                        required=True,
                        help='Path to training data _file_list.txt')
    parser.add_argument('-v',
                        '--valid_data',
                        type=str,
                        required=True,
                        help='Path to validation data _file_list.txt')
    parser.add_argument('--dropout_rate',
                        type=float,
                        required=False,
                        default=0.5,
                        help='Dropout rate')
    parser.add_argument('--num_dense_features',
                        type=int,
                        required=False,
                        default=13,
                        help='Number of dense features')
    parser.add_argument('--num_sparse_features',
                        type=int,
                        required=False,
                        default=26,
                        help='Number of sparse features')
    parser.add_argument('--lr',
                        type=float,
                        required=False,
                        default=0.001,
                        help='Learning rate')
    
    parser.add_argument('-i',
                        '--max_iter',
                        type=int,
                        required=False,
                        default=0,
                        help='Number of training iterations')
    parser.add_argument('--num_epochs',
                        type=int,
                        required=False,
                        default=1,
                        help='Number of training epochs')
    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        required=False,
                        default=16384,
                        help='Batch size')
    parser.add_argument('-s',
                        '--snapshot',
                        type=int,
                        required=False,
                        default=10000,
                        help='Saves a model snapshot after given number of iterations')
    parser.add_argument('--gpus',
                        type=str,
                        required=False,
                        default="[[0]]",
                        help='GPU devices to use for Preprocessing')
    parser.add_argument('-r',
                        '--eval_interval',
                        type=int,
                        required=False,
                        default=1000,
                        help='Run evaluation after given number of iterations')
    parser.add_argument('--display_interval',
                        type=int,
                        required=False,
                        default=100,
                        help='Display progress after given number of iterations')
    parser.add_argument('--slot_size_array',
                        type=str,
                        required=True,
                        help='Categorical variables cardinalities')
    parser.add_argument('--workspace_size_per_gpu',
                        type=int,
                        required=False,
                        default=1000,
                        help='Workspace size per gpu in MB')
    parser.add_argument('--num_workers',
                        type=int,
                        required=False,
                        default=12,
                        help='Number of workers')


    args = parser.parse_args()

    return args  

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')

    args = parse_args()
    args.gpus = json.loads(args.gpus)
    args.slot_size_array = json.loads(args.slot_size_array)

    logging.info(f"Args: {args}")
    start_time = time.time()
    logging.info("Starting training")

    main(args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Training completed. Elapsed time: {}".format(elapsed_time))
