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
import shutil


from trainer.model import create_model

MODEL_PREFIX = 'deepfm'
SNAPSHOT_DIR = 'snapshots'
GRAPH_DIR = 'graph'
MODEL_PARAMETERS_DIR = 'parameters'


def save_model(model, model_dir):
    """Saves model graph and model parameters."""
    
    graph_path = os.path.join(model_dir, GRAPH_DIR)
    if os.path.isdir(graph_path):
        shutil.rmtree(graph_path)
    os.makedirs(graph_path)
                     
    graph_path = os.path.join(graph_path, f'{MODEL_PREFIX}.json')
    logging.info('Saving model graph to: {}'.format(graph_path))  
    
    model.graph_to_json(graph_config_file=graph_path)
   
    parameters_path = os.path.join(model_dir, MODEL_PARAMETERS_DIR, MODEL_PREFIX)
    logging.info('Saving model parameters to: {}'.format(parameters_path)) 
    model.save_params_to_files(prefix=parameters_path)
    

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
    
    logging.info('Starting model fitting')
    model.fit(
        num_epochs=args.num_epochs,
        max_iter=args.max_iter,
        display=args.display_interval, 
        eval_interval=args.eval_interval, 
        snapshot=args.snapshot_interval, 
        snapshot_prefix=os.path.join(args.model_dir, SNAPSHOT_DIR, MODEL_PREFIX))
    
    logging.info('Saving model')
    save_model(model, args.model_dir)
    

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
                        '--snapshot_interval',
                        type=int,
                        required=False,
                        default=10000,
                        help='Saves a model snapshot after given number of iterations')
    parser.add_argument('--model_dir',
                        type=str,
                        required=False,
                        default="/tmp/model",
                        help='A base directory for snaphosts and saved model.')  
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
