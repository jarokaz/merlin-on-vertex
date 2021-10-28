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
"""Exporting Triton ensemble model."""

import os
import json

import nvtabular as nvt
from nvtabular.inference.triton import export_hugectr_ensemble

from pathlib import Path


NUM_SLOTS = 26
MAX_NNZ = 2
NUM_OUTPUTS = 1
EMBEDDING_VECTOR_SIZE = 11
MAX_BATCH_SIZE = 64

MODEL_PREFIX = 'deepfm'
MODEL_REGISTRY_PATH = "/models"
HUGECTR_CONFIG_FILENAME = "ps.json"


def create_hugectr_backend_config(
    model_path):
    
    p = Path(model_path)
    model_version = p.parts[-1]
    model_name = p.parts[-2]
    model_path_in_registry = os.path.join(MODEL_REGISTRY_PATH, model_name, model_version)
    
    dense_pattern=f'{model_name}_dense_*.model'
    dense_path = [os.path.join(model_path_in_registry, path.name) 
                  for path in p.glob(dense_pattern)][0]
    sparse_pattern=f'{model_name}[0-9]_sparse_*.model'
    sparse_paths = [os.path.join(model_path_in_registry, path.name) 
                    for path in p.glob(sparse_pattern)]   
    network_file = os.path.join(model_path_in_registry, f'{model_name}.json')

    config_dict = dict()
    config_dict['supportlonglong'] = True
    model_config = dict()
    model_config['model'] = model_name
    model_config['sparse_files'] = sparse_paths
    model_config['dense_file'] = dense_path
    model_config['network_file'] = network_file
    config_dict['models'] = [model_config]
    
    return config_dict
    

def export_ensemble(
    workflow_path,
    saved_model_path,
    output_path,
    categorical_columns,
    continuous_columns,
    label_columns):
          
    workflow = nvt.Workflow.load(workflow_path)
    
    hugectr_params = dict()
    graph_filename = f'{MODEL_PREFIX}.json'
    hugectr_params["config"] = os.path.join(
        MODEL_REGISTRY_PATH,
        MODEL_PREFIX,
        "1",
        graph_filename)
    
    hugectr_params["slots"] = NUM_SLOTS
    hugectr_params["max_nnz"] = MAX_NNZ
    hugectr_params["embedding_vector_size"] = EMBEDDING_VECTOR_SIZE
    hugectr_params["n_outputs"] = NUM_OUTPUTS
    
    export_hugectr_ensemble(
        workflow=workflow,
        hugectr_model_path=saved_model_path,
        hugectr_params=hugectr_params,
        name=MODEL_PREFIX,
        output_path=output_path,
        label_columns=label_columns,
        cats=categorical_columns,
        conts=continuous_columns,
        max_batch_size=MAX_BATCH_SIZE,
    )
    
    hugectr_backend_config = create_hugectr_backend_config(
        model_path=os.path.join(output_path, MODEL_PREFIX, '1'))
    
    with open(os.path.join(output_path, HUGECTR_CONFIG_FILENAME), 'w') as f:
        json.dump(hugectr_backend_config, f)
    

    
 
   