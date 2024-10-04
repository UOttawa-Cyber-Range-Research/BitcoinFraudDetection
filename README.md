
# BitcoinFraudDetection

The repository is the codebase for Bitcoing Fraud Detection done by Karanjot Singh Saggu under Professor Paula Branco and Professor Guy-Vincent Jourdan.

The research aims to undestand what Bitcoin databases are and how they can be used with Graph Neural Networks to detect Fraudulent Activity.

The database of the code is a private database. You can open a request in the issue tab or send an email to ksagg020@uottawa.ca to get details on how to get the access to the database.

## Sampling methods
The code for the sampling methods is available in two differnt locations.
1.  If you want to use the jupyter files refer the the searchNoteBooks folder to access the files and run them from there.
2. The other option is to use the seachScripts folder to access file and run the sampling methods from (recommended). The code for doing the same is.

```
python BitcoinFraudDetection/{ds_name}.py 
```


## Training the models
The code base is divided into 2 differnt task that you can run and evaluate.

1. The normal train pipeline for training one model at a time. You can run the script using the following command

```
python BitcoinFraudDetection/train.py {model_name} -data_path BitcoinFraudDetection/data/{ds_name}\
-features_dir BitcoinFraudDetection/data/table\
-class_weight {class_weight} -rwpe {rwpe_value} -norm {norm_value} 

```

2. The normal train pipeline for getting the smoothing metrics with the RGGCN model. You can run the script using the following command

```
python train_compute_IIG_GDR_rggcn.py {model_name} -data_path BitcoinFraudDetection/data/{ds_name}\
-features_dir BitcoinFraudDetection/data/table\
-class_weight {class_weight} -rwpe {rwpe_value} -norm {norm_value} 

```

The values that each variable can take are"

``` 
model_name: {dgcn, gat, gcn, gps, gs, rggcn}
```

``` 
ds_name: {datasetRFS, datasetBFS, datasetFS, datasetBFRON, datasetMHRW}
```

``` 
class_weight: deafult to 117 but can be recalculated using positve and negative samples in the chosen sampling method
```

``` 
rwpe_value: {true or false}
```

``` 
norm_value: {BN, GN}
```

3. To run the lgbm code you can use the notebooks inside the lgbm folder and directly run the script to train the model and get the results.

## Visualizations
The visualization code is seperated into two file.

1. For result Visulaiztions we use the result_visualization.ipynb or result_visualization_seperate.ipynb file.
2. For smoothness metric analysis run the gdr_iig_analysis.ipynb file to visualize both of these metrics.

## Runtime Analysis of different sampling methods
The following run time was calculated for each sampling method on the same graph across 100 fraudulent nodes

| Sampling Method  | Runtime (ms) |
| ------------- | ------------- |
| BFS  | 538.64 |
| RFS  | 89.74  |
| FS  | 142.78  |
| MHRW  | 70.31  |
| BFRON  | 444.28  |
