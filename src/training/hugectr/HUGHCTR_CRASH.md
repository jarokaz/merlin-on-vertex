# HUGECTR debugging

To replicate a crash during training on a `a2-highgpu-8g`.

## Provision a Vertex AI notebook instance 

Use the following settings:

- Use `a2-highgpu-8g` machine type
- Use TensorFlow Enterprise 2.6 image
- Set boot disk to 500GB
- Set data disk to 3000GB
- Install GPU driver automatically

# Clone this repo

Log on to JupyterLab and open a Jupyter terminal. 

```
cd
git clone https://github.com/jarokaz/merlin-on-vertex.git

```



## Get the Criteo dataset

Download the preprocessed Criteo dataset.

```
cd
mkdir data
cd data
gsutil -m cp -r gs://workshop-datasets/criteo_processed_parquet .
```

## Run tests

First run the test using 4 GPUs. This should work.

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-on-vertex/src/vertex_training/hugectr:/src \
-v /home/jupyter/data:/criteo_data \
-w /src \
nvcr.io/nvidia/merlin/merlin-training:21.09 \
python -m trainer.train \
--num_epochs 0 \
--max_iter 50000 \
--eval_interval=5000 \
--batchsize=8192 \
--snapshot=0 \
--train_data=/criteo_data/criteo_processed_parquet/train/_file_list.txt  \
--valid_data=/criteo_data/criteo_processed_parquet/valid/_file_list.txt  \
--display_interval=500 \
--workspace_size_per_gpu=61 \
--slot_size_array="[18792578, 35176, 17091, 7383, 20154, 4, 7075, 1403, 63, 12687136, 1054830, 297377, 11, 2209, 10933, 113, 4, 972, 15, 19550853, 5602712, 16779972, 375290, 12292, 101, 35]" \
--gpus="[[0,1,2,3]]"
```

Then run the test using 8 GPUs. This crashes.

```
docker run -it --rm --gpus all --cap-add SYS_NICE \
-v /home/jupyter/merlin-on-vertex/src/vertex_training/hugectr:/src \
-v /home/jupyter/data:/criteo_data \
-w /src \
nvcr.io/nvidia/merlin/merlin-training:21.09 \
python -m trainer.train \
--num_epochs 0 \
--max_iter 50000 \
--eval_interval=5000 \
--batchsize=16384 \
--snapshot=0 \
--train_data=/criteo_data/criteo_processed_parquet/train/_file_list.txt  \
--valid_data=/criteo_data/criteo_processed_parquet/valid/_file_list.txt  \
--display_interval=500 \
--workspace_size_per_gpu=61 \
--slot_size_array="[18792578, 35176, 17091, 7383, 20154, 4, 7075, 1403, 63, 12687136, 1054830, 297377, 11, 2209, 10933, 113, 4, 972, 15, 19550853, 5602712, 16779972, 375290, 12292, 101, 35]" \
--gpus="[[0,1,2,3,4,5,6,7]]"
```


The logs generated during the crash.

```
05-10-21 19:12:54 - Args: Namespace(batchsize=16384, display_interval=500, eval_interval=5000, gpus=[[0, 1, 2, 3, 4, 5, 6, 7]], lr=0.001, max_iter=50000, num_epochs=0, slot_size_array=[18792578, 35176, 17091, 7383, 20154, 4, 7075, 1403, 63, 12687136, 1054830, 297377, 11, 2209, 10933, 113, 4, 972, 15, 19550853, 5602712, 16779972, 375290, 12292, 101, 35], snapshot=0, train_data='/criteo_data/criteo_processed_parquet/train/_file_list.txt', valid_data='/criteo_data/criteo_processed_parquet/valid/_file_list.txt', workspace_size_per_gpu=61)
05-10-21 19:12:54 - Starting training
HugeCTR Version: 3.2.0
====================================================Model Init=====================================================
[05d19h12m54s][HUGECTR][INFO]: Global seed is 1479669930
[05d19h12m55s][HUGECTR][INFO]: Device to NUMA mapping:
  GPU 0 ->  node 0
  GPU 1 ->  node 0
  GPU 2 ->  node 0
  GPU 3 ->  node 0
  GPU 4 ->  node 1
  GPU 5 ->  node 1
  GPU 6 ->  node 1
  GPU 7 ->  node 1

[05d19h13m27s][HUGECTR][INFO]: Start all2all warmup
[05d19h13m45s][HUGECTR][INFO]: End all2all warmup
[05d19h13m45s][HUGECTR][INFO]: Using All-reduce algorithm NCCL
Device 0: A100-SXM4-40GB
Device 1: A100-SXM4-40GB
Device 2: A100-SXM4-40GB
Device 3: A100-SXM4-40GB
Device 4: A100-SXM4-40GB
Device 5: A100-SXM4-40GB
Device 6: A100-SXM4-40GB
Device 7: A100-SXM4-40GB
[05d19h13m45s][HUGECTR][INFO]: num of DataReader workers: 8
[05d19h13m45s][HUGECTR][INFO]: Vocabulary size: 75255782
[05d19h13m45s][HUGECTR][INFO]: max_vocabulary_size_per_gpu_=19855613
[05d19h13m45s][HUGECTR][INFO]: All2All Warmup Start
[05d19h13m45s][HUGECTR][INFO]: All2All Warmup End
===================================================Model Compile===================================================
[05d19h13m56s][HUGECTR][INFO]: gpu0 start to init embedding of slot0 , slot_size=18792578, key_offset=0, value_index_offset=0
[05d19h13m56s][HUGECTR][INFO]: gpu0 start to init embedding of slot8 , slot_size=63, key_offset=18880864, value_index_offset=18792578
[05d19h13m56s][HUGECTR][INFO]: gpu0 start to init embedding of slot16 , slot_size=4, key_offset=32933536, value_index_offset=18792641
[05d19h13m56s][HUGECTR][INFO]: gpu0 start to init embedding of slot24 , slot_size=101, key_offset=75255646, value_index_offset=18792645
[05d19h13m56s][HUGECTR][INFO]: gpu1 start to init embedding of slot1 , slot_size=35176, key_offset=18792578, value_index_offset=0
[05d19h13m56s][HUGECTR][INFO]: gpu1 start to init embedding of slot9 , slot_size=12687136, key_offset=18880927, value_index_offset=35176
[05d19h13m56s][HUGECTR][INFO]: gpu1 start to init embedding of slot17 , slot_size=972, key_offset=32933540, value_index_offset=12722312
[05d19h13m56s][HUGECTR][INFO]: gpu1 start to init embedding of slot25 , slot_size=35, key_offset=75255747, value_index_offset=12723284
[05d19h13m56s][HUGECTR][INFO]: gpu2 start to init embedding of slot2 , slot_size=17091, key_offset=18827754, value_index_offset=0
[05d19h13m56s][HUGECTR][INFO]: gpu2 start to init embedding of slot10 , slot_size=1054830, key_offset=31568063, value_index_offset=17091
[05d19h13m56s][HUGECTR][INFO]: gpu2 start to init embedding of slot18 , slot_size=15, key_offset=32934512, value_index_offset=1071921
[05d19h13m56s][HUGECTR][INFO]: gpu3 start to init embedding of slot3 , slot_size=7383, key_offset=18844845, value_index_offset=0
[05d19h13m56s][HUGECTR][INFO]: gpu3 start to init embedding of slot11 , slot_size=297377, key_offset=32622893, value_index_offset=7383
[05d19h13m56s][HUGECTR][INFO]: gpu3 start to init embedding of slot19 , slot_size=19550853, key_offset=32934527, value_index_offset=304760
[05d19h13m56s][HUGECTR][INFO]: gpu4 start to init embedding of slot4 , slot_size=20154, key_offset=18852228, value_index_offset=0
[05d19h13m56s][HUGECTR][INFO]: gpu4 start to init embedding of slot12 , slot_size=11, key_offset=32920270, value_index_offset=20154
[05d19h13m56s][HUGECTR][INFO]: gpu4 start to init embedding of slot20 , slot_size=5602712, key_offset=52485380, value_index_offset=20165
[05d19h13m56s][HUGECTR][INFO]: gpu5 start to init embedding of slot5 , slot_size=4, key_offset=18872382, value_index_offset=0
[05d19h13m56s][HUGECTR][INFO]: gpu5 start to init embedding of slot13 , slot_size=2209, key_offset=32920281, value_index_offset=4
[05d19h13m56s][HUGECTR][INFO]: gpu5 start to init embedding of slot21 , slot_size=16779972, key_offset=58088092, value_index_offset=2213
[05d19h13m56s][HUGECTR][INFO]: gpu6 start to init embedding of slot6 , slot_size=7075, key_offset=18872386, value_index_offset=0
[05d19h13m56s][HUGECTR][INFO]: gpu6 start to init embedding of slot14 , slot_size=10933, key_offset=32922490, value_index_offset=7075
[05d19h13m56s][HUGECTR][INFO]: gpu6 start to init embedding of slot22 , slot_size=375290, key_offset=74868064, value_index_offset=18008
[05d19h13m56s][HUGECTR][INFO]: gpu7 start to init embedding of slot7 , slot_size=1403, key_offset=18879461, value_index_offset=0
[05d19h13m56s][HUGECTR][INFO]: gpu7 start to init embedding of slot15 , slot_size=113, key_offset=32933423, value_index_offset=1403
[05d19h13m56s][HUGECTR][INFO]: gpu7 start to init embedding of slot23 , slot_size=12292, key_offset=75243354, value_index_offset=1516
[05d19h13m56s][HUGECTR][INFO]: gpu0 init embedding done
[05d19h13m56s][HUGECTR][INFO]: gpu1 init embedding done
[05d19h13m56s][HUGECTR][INFO]: gpu2 init embedding done
[05d19h13m56s][HUGECTR][INFO]: gpu3 init embedding done
[05d19h13m56s][HUGECTR][INFO]: gpu4 init embedding done
[05d19h13m56s][HUGECTR][INFO]: gpu5 init embedding done
[05d19h13m56s][HUGECTR][INFO]: gpu6 init embedding done
[05d19h13m56s][HUGECTR][INFO]: gpu7 init embedding done
[05d19h13m56s][HUGECTR][INFO]: Starting AUC NCCL warm-up
[05d19h13m56s][HUGECTR][INFO]: Warm-up done
===================================================Model Summary===================================================
Label                                   Dense                         Sparse                        
label                                   dense                          data1                         
(None, 1)                               (None, 13)                              
------------------------------------------------------------------------------------------------------------------
Layer Type                              Input Name                    Output Name                   Output Shape                  
------------------------------------------------------------------------------------------------------------------
LocalizedSlotSparseEmbeddingHash        data1                         sparse_embedding1             (None, 26, 11)                
Reshape                                 sparse_embedding1             reshape1                      (None, 11)                    
Slice                                   reshape1                      slice11,slice12                                             
Reshape                                 slice11                       reshape2                      (None, 260)                   
Reshape                                 slice12                       reshape3                      (None, 26)                    
Slice                                   dense                         slice21,slice22                                             
WeightMultiply                          slice21                       weight_multiply1              (None, 130)                   
WeightMultiply                          slice22                       weight_multiply2              (None, 13)                    
Concat                                  reshape2,weight_multiply1     concat1                       (None, 390)                   
Slice                                   concat1                       slice31,slice32                                             
InnerProduct                            slice31                       fc1                           (None, 400)                   
ReLU                                    fc1                           relu1                         (None, 400)                   
Dropout                                 relu1                         dropout1                      (None, 400)                   
InnerProduct                            dropout1                      fc2                           (None, 400)                   
ReLU                                    fc2                           relu2                         (None, 400)                   
Dropout                                 relu2                         dropout2                      (None, 400)                   
InnerProduct                            dropout2                      fc3                           (None, 400)                   
ReLU                                    fc3                           relu3                         (None, 400)                   
Dropout                                 relu3                         dropout3                      (None, 400)                   
InnerProduct                            dropout3                      fc4                           (None, 1)                     
FmOrder2                                slice32                       fmorder2                      (None, 10)                    
ReduceSum                               fmorder2                      reducesum1                    (None, 1)                     
Concat                                  reshape3,weight_multiply2     concat2                       (None, 39)                    
ReduceSum                               concat2                       reducesum2                    (None, 1)                     
Add                                     fc4,reducesum1,reducesum2     add                           (None, 1)                     
BinaryCrossEntropyLoss                  add,label                     loss                                                        
------------------------------------------------------------------------------------------------------------------
=====================================================Model Fit=====================================================
[50d19h13m56s][HUGECTR][INFO]: Use non-epoch mode with number of iterations: 50000
[50d19h13m56s][HUGECTR][INFO]: Training batchsize: 16384, evaluation batchsize: 16384
[50d19h13m56s][HUGECTR][INFO]: Evaluation interval: 5000, snapshot interval: 0
[50d19h13m56s][HUGECTR][INFO]: Sparse embedding trainable: 1, dense network trainable: 1
[50d19h13m56s][HUGECTR][INFO]: Use mixed precision: 0, scaler: 1.000000, use cuda graph: 1
[50d19h13m56s][HUGECTR][INFO]: lr: 0.001000, warmup_steps: 1, decay_start: 0, decay_steps: 1, decay_power: 2.000000, end_lr: 0.000000
[50d19h13m56s][HUGECTR][INFO]: Training source file: /criteo_data/criteo_processed_parquet/train/_file_list.txt
[50d19h13m56s][HUGECTR][INFO]: Evaluation source file: /criteo_data/criteo_processed_parquet/valid/_file_list.txt
[HCDEBUG][ERROR] Runtime error: an illegal memory access was encountered /var/tmp/HugeCTR/HugeCTR/include/data_readers/data_collector.hpp:236 

Traceback (most recent call last):
  File "/usr/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/src/trainer/train.py", line 135, in <module>
    main(args)
  File "/src/trainer/train.py", line 43, in main
    model.fit(
RuntimeError: [HCDEBUG][ERROR] Runtime error: an illegal memory access was encountered /var/tmp/HugeCTR/HugeCTR/include/data_readers/data_collector.hpp:236 

[HCDEBUG][ERROR] Runtime error: an illegal memory access was encountered /var/tmp/HugeCTR/HugeCTR/pybind/model.cpp:406 

terminate called after throwing an instance of 'HugeCTR::internal_runtime_error'
  what():  [HCDEBUG][ERROR] Runtime error: an illegal memory access was encountered /var/tmp/HugeCTR/HugeCTR/src/metrics.cu:497 

[fa3489fe7e0b:00001] *** Process received signal ***
[fa3489fe7e0b:00001] Signal: Aborted (6)
[fa3489fe7e0b:00001] Signal code:  (-6)
[fa3489fe7e0b:00001] [ 0] /usr/lib/x86_64-linux-gnu/libc.so.6(+0x46210)[0x7f125d991210]
[fa3489fe7e0b:00001] [ 1] /usr/lib/x86_64-linux-gnu/libc.so.6(gsignal+0xcb)[0x7f125d99118b]
[fa3489fe7e0b:00001] [ 2] /usr/lib/x86_64-linux-gnu/libc.so.6(abort+0x12b)[0x7f125d970859]
[fa3489fe7e0b:00001] [ 3] /usr/lib/x86_64-linux-gnu/libstdc++.so.6(+0x9e911)[0x7f124385d911]
[fa3489fe7e0b:00001] [ 4] /usr/lib/x86_64-linux-gnu/libstdc++.so.6(+0xaa38c)[0x7f124386938c]
[fa3489fe7e0b:00001] [ 5] /usr/lib/x86_64-linux-gnu/libstdc++.so.6(+0xa9369)[0x7f1243868369]
[fa3489fe7e0b:00001] [ 6] /usr/lib/x86_64-linux-gnu/libstdc++.so.6(__gxx_personality_v0+0x2a1)[0x7f1243868d21]
[fa3489fe7e0b:00001] [ 7] /usr/lib/x86_64-linux-gnu/libgcc_s.so.1(+0x10bef)[0x7f1243772bef]
[fa3489fe7e0b:00001] [ 8] /usr/lib/x86_64-linux-gnu/libgcc_s.so.1(_Unwind_RaiseException+0x331)[0x7f1243773281]
[fa3489fe7e0b:00001] [ 9] /usr/lib/x86_64-linux-gnu/libstdc++.so.6(__cxa_throw+0x3c)[0x7f124386969c]
[fa3489fe7e0b:00001] [10] /usr/local/hugectr/lib/libhuge_ctr_shared.so(+0x20bf9a)[0x7f125a721f9a]
[fa3489fe7e0b:00001] [11] /usr/local/hugectr/lib/libhuge_ctr_shared.so(_ZN7HugeCTR7metrics3AUCIfED1Ev+0x86)[0x7f125acdccd6]
[fa3489fe7e0b:00001] [12] /usr/local/hugectr/lib/libhuge_ctr_shared.so(_ZN7HugeCTR7metrics3AUCIfED0Ev+0xd)[0x7f125acdcdcd]
[fa3489fe7e0b:00001] [13] /usr/local/hugectr/lib/libhuge_ctr_shared.so(_ZN7HugeCTR5ModelD1Ev+0x127)[0x7f125ae66327]
[fa3489fe7e0b:00001] [14] /usr/local/hugectr/lib/hugectr.so(+0x4b956)[0x7f125d1df956]
[fa3489fe7e0b:00001] [15] /usr/local/hugectr/lib/hugectr.so(_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv+0x48)[0x7f125d1f5528]
[fa3489fe7e0b:00001] [16] /usr/local/hugectr/lib/hugectr.so(+0x61f34)[0x7f125d1f5f34]
[fa3489fe7e0b:00001] [17] /usr/local/hugectr/lib/hugectr.so(+0xce577)[0x7f125d262577]
[fa3489fe7e0b:00001] [18] /usr/local/hugectr/lib/hugectr.so(+0xcf463)[0x7f125d263463]
[fa3489fe7e0b:00001] [19] python[0x5eb950]
[fa3489fe7e0b:00001] [20] python[0x5434c8]
[fa3489fe7e0b:00001] [21] python[0x54351a]
[fa3489fe7e0b:00001] [22] python[0x54351a]
[fa3489fe7e0b:00001] [23] python[0x54351a]
[fa3489fe7e0b:00001] [24] python(PyDict_SetItemString+0x536)[0x5d08b6]
[fa3489fe7e0b:00001] [25] python(PyImport_Cleanup+0x79)[0x684429]
[fa3489fe7e0b:00001] [26] python(Py_FinalizeEx+0x7f)[0x67f6bf]
[fa3489fe7e0b:00001] [27] python(Py_RunMain+0x32d)[0x6b6f7d]
[fa3489fe7e0b:00001] [28] python(Py_BytesMain+0x2d)[0x6b71ed]
[fa3489fe7e0b:00001] [29] /usr/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xf3)[0x7f125d9720b3]
[fa3489fe7e0b:00001] *** End of error message ***
[fa3489fe7e0b:1    :0:1] Caught signal 11 (Segmentation fault: Sent by the kernel at address (nil))
==== backtrace (tid:      1) ====
 0  /usr/local/ucx/lib/libucs.so.0(ucs_handle_error+0x2a4) [0x7f11fdc3ed24]
 1  /usr/local/ucx/lib/libucs.so.0(+0x27eff) [0x7f11fdc3eeff]
 2  /usr/local/ucx/lib/libucs.so.0(+0x28234) [0x7f11fdc3f234]
 3  /usr/lib/x86_64-linux-gnu/libpthread.so.0(+0x153c0) [0x7f125d93d3c0]
 4  /usr/lib/x86_64-linux-gnu/libc.so.6(abort+0x213) [0x7f125d970941]
 5  /usr/lib/x86_64-linux-gnu/libstdc++.so.6(+0x9e911) [0x7f124385d911]
 6  /usr/lib/x86_64-linux-gnu/libstdc++.so.6(+0xaa38c) [0x7f124386938c]
 7  /usr/lib/x86_64-linux-gnu/libstdc++.so.6(+0xa9369) [0x7f1243868369]
 8  /usr/lib/x86_64-linux-gnu/libstdc++.so.6(__gxx_personality_v0+0x2a1) [0x7f1243868d21]
 9  /usr/lib/x86_64-linux-gnu/libgcc_s.so.1(+0x10bef) [0x7f1243772bef]
10  /usr/lib/x86_64-linux-gnu/libgcc_s.so.1(_Unwind_RaiseException+0x331) [0x7f1243773281]
11  /usr/lib/x86_64-linux-gnu/libstdc++.so.6(__cxa_throw+0x3c) [0x7f124386969c]
12  /usr/local/hugectr/lib/libhuge_ctr_shared.so(+0x20bf9a) [0x7f125a721f9a]
13  /usr/local/hugectr/lib/libhuge_ctr_shared.so(_ZN7HugeCTR7metrics3AUCIfED1Ev+0x86) [0x7f125acdccd6]
14  /usr/local/hugectr/lib/libhuge_ctr_shared.so(_ZN7HugeCTR7metrics3AUCIfED0Ev+0xd) [0x7f125acdcdcd]
15  /usr/local/hugectr/lib/libhuge_ctr_shared.so(_ZN7HugeCTR5ModelD1Ev+0x127) [0x7f125ae66327]
16  /usr/local/hugectr/lib/hugectr.so(+0x4b956) [0x7f125d1df956]
17  /usr/local/hugectr/lib/hugectr.so(_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv+0x48) [0x7f125d1f5528]
18  /usr/local/hugectr/lib/hugectr.so(+0x61f34) [0x7f125d1f5f34]
19  /usr/local/hugectr/lib/hugectr.so(+0xce577) [0x7f125d262577]
20  /usr/local/hugectr/lib/hugectr.so(+0xcf463) [0x7f125d263463]
21  python() [0x5eb950]
22  python() [0x5434c8]
23  python() [0x54351a]
24  python() [0x54351a]
25  python() [0x54351a]
26  python(PyDict_SetItemString+0x536) [0x5d08b6]
27  python(PyImport_Cleanup+0x79) [0x684429]
28  python(Py_FinalizeEx+0x7f) [0x67f6bf]
29  python(Py_RunMain+0x32d) [0x6b6f7d]
30  python(Py_BytesMain+0x2d) [0x6b71ed]
31  /usr/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xf3) [0x7f125d9720b3]
32  python(_start+0x2e) [0x5f96de]
=================================
[fa3489fe7e0b:00001] *** Process received signal ***
[fa3489fe7e0b:00001] Signal: Segmentation fault (11)
[fa3489fe7e0b:00001] Signal code:  (-6)
[fa3489fe7e0b:00001] Failing at address: 0x1
[fa3489fe7e0b:00001] [ 0] /usr/lib/x86_64-linux-gnu/libpthread.so.0(+0x153c0)[0x7f125d93d3c0]
[fa3489fe7e0b:00001] [ 1] /usr/lib/x86_64-linux-gnu/libc.so.6(abort+0x213)[0x7f125d970941]
[fa3489fe7e0b:00001] [ 2] /usr/lib/x86_64-linux-gnu/libstdc++.so.6(+0x9e911)[0x7f124385d911]
[fa3489fe7e0b:00001] [ 3] /usr/lib/x86_64-linux-gnu/libstdc++.so.6(+0xaa38c)[0x7f124386938c]
[fa3489fe7e0b:00001] [ 4] /usr/lib/x86_64-linux-gnu/libstdc++.so.6(+0xa9369)[0x7f1243868369]
[fa3489fe7e0b:00001] [ 5] /usr/lib/x86_64-linux-gnu/libstdc++.so.6(__gxx_personality_v0+0x2a1)[0x7f1243868d21]
[fa3489fe7e0b:00001] [ 6] /usr/lib/x86_64-linux-gnu/libgcc_s.so.1(+0x10bef)[0x7f1243772bef]
[fa3489fe7e0b:00001] [ 7] /usr/lib/x86_64-linux-gnu/libgcc_s.so.1(_Unwind_RaiseException+0x331)[0x7f1243773281]
[fa3489fe7e0b:00001] [ 8] /usr/lib/x86_64-linux-gnu/libstdc++.so.6(__cxa_throw+0x3c)[0x7f124386969c]
[fa3489fe7e0b:00001] [ 9] /usr/local/hugectr/lib/libhuge_ctr_shared.so(+0x20bf9a)[0x7f125a721f9a]
[fa3489fe7e0b:00001] [10] /usr/local/hugectr/lib/libhuge_ctr_shared.so(_ZN7HugeCTR7metrics3AUCIfED1Ev+0x86)[0x7f125acdccd6]
[fa3489fe7e0b:00001] [11] /usr/local/hugectr/lib/libhuge_ctr_shared.so(_ZN7HugeCTR7metrics3AUCIfED0Ev+0xd)[0x7f125acdcdcd]
[fa3489fe7e0b:00001] [12] /usr/local/hugectr/lib/libhuge_ctr_shared.so(_ZN7HugeCTR5ModelD1Ev+0x127)[0x7f125ae66327]
[fa3489fe7e0b:00001] [13] /usr/local/hugectr/lib/hugectr.so(+0x4b956)[0x7f125d1df956]
[fa3489fe7e0b:00001] [14] /usr/local/hugectr/lib/hugectr.so(_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv+0x48)[0x7f125d1f5528]
[fa3489fe7e0b:00001] [15] /usr/local/hugectr/lib/hugectr.so(+0x61f34)[0x7f125d1f5f34]
[fa3489fe7e0b:00001] [16] /usr/local/hugectr/lib/hugectr.so(+0xce577)[0x7f125d262577]
[fa3489fe7e0b:00001] [17] /usr/local/hugectr/lib/hugectr.so(+0xcf463)[0x7f125d263463]
[fa3489fe7e0b:00001] [18] python[0x5eb950]
[fa3489fe7e0b:00001] [19] python[0x5434c8]
[fa3489fe7e0b:00001] [20] python[0x54351a]
[fa3489fe7e0b:00001] [21] python[0x54351a]
[fa3489fe7e0b:00001] [22] python[0x54351a]
[fa3489fe7e0b:00001] [23] python(PyDict_SetItemString+0x536)[0x5d08b6]
[fa3489fe7e0b:00001] [24] python(PyImport_Cleanup+0x79)[0x684429]
[fa3489fe7e0b:00001] [25] python(Py_FinalizeEx+0x7f)[0x67f6bf]
[fa3489fe7e0b:00001] [26] python(Py_RunMain+0x32d)[0x6b6f7d]
[fa3489fe7e0b:00001] [27] python(Py_BytesMain+0x2d)[0x6b71ed]
[fa3489fe7e0b:00001] [28] /usr/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xf3)[0x7f125d9720b3]
[fa3489fe7e0b:00001] [29] python(_start+0x2e)[0x5f96de]
[fa3489fe7e0b:00001] *** End of error message ***
```
