The project structure is as follows：

-StarG2N: Contains the source code for StarG2N, as well as experimental results and model files on four read-world datasets.
         Metrics.py is the file of Metrics.
         model_v6.py is the model file of StarG2N on PeMS03, PeMS04, and PeMS07.
         model_v7.py is the model file of StarG2N on PeMS08.
         pems03test.py is the test file of StarG2N on PeMS03.
         pems04test.py is the test file of StarG2N on PeMS04.
         pems07test.py is the test file of StarG2N on PeMS07.
         pems08test.py is the test file of StarG2N on PeMS08.
         traffic_dataset_v4.py is the data processing file of StarG2N on PeMS04, PeMS07, and PeMS08.
         traffic_dataset_v5.py is the data processing file of StarG2N on PeMS03.
         result_test.ipynb can load the model and output the results directly.

-Baseline: Contains the source code for Attention, Chebnet, GRU, T-GCN, as well as experimental results and model files on four read-world datasets.
         Baseline_traffic_prediction.py is the test file of baseline models.
         Chebnet_model.py is the model file of Chebnet.
         GRU_model.py is the model file of GRU.
         Multi_head_attention.py is the model file of Multi_head_attention.
         T_GCN_model.py is the model file of T_GCN.
         traffice_dataset_normal.py is the data processing file of baseline models without the construction of local spatial-temproal graph.

-data: Contains data from four datasets, including PeMS03, PeMS04, PeMS07, and PeMS08.

-visualization: Contains source code for visualization of experimental results.
         visualization.ipynb is the visualization file.

Note: 
         All files with ".pt" suffix are pre-stored model files and can be used directly by users.
         The program may contain dataset loading and result storage at runtime, so users need to pay attention to the path issue, otherwise the program will not run successfully.

