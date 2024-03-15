import os
import gc
import pandas as pd
import networkx as nx
from lib.store import Store
from tqdm import tqdm
from collections import Counter
from typing import Dict
from sklearn.preprocessing import MinMaxScaler
import random

import warnings
warnings.simplefilter("ignore")

from lib.naming_corrections import (
    FEATURE_COLUMNS_OTHERS,
    FEATURES_NAMES_FROM_NEW_CACHE,
    FEATURES_NAMES_FROM_PRELOADED_CACHE,
    TABLES_COLUMNS_DEFAULT_LEGACY,
    TABLES_V5_2_V4_RENAME_LEGACY,
)

FEATURE_COLUMNS = FEATURES_NAMES_FROM_PRELOADED_CACHE + FEATURE_COLUMNS_OTHERS
TABLES_V5_2_V4_RENAME = TABLES_V5_2_V4_RENAME_LEGACY
TABLES_COLUMNS_DEFAULT = TABLES_COLUMNS_DEFAULT_LEGACY


# function for mhrw
def generic_mhrw_edges(G, source, label_dict, neighbors=None,
                       depth_limit=None, sort_neighbors=None,
                       factor=100):
    """
    Iterate over edges in a MHR search.
    """

    # Define the ratio 
    ratio = {0: 0, 1: 0}
    visited = {source}

    # Define the depth limit
    if depth_limit is None:
        depth_limit = len(G)

    # No need for a queue
    node_list = set()
    parent_node = source
    parent_neighbors = list(neighbors(source))
    node_list.update(parent_neighbors)

    # Find the degree of the parent node
    degree_parent = G.degree(source)

    # Maintain the ratio
    ratio[label_dict[source]] += 1

    # Placeholder for collected edges
    edges = []
    set_all_added_nodes = set()

    # Define the logic for MHRW
    while (ratio[0]/ratio[1]) < factor:
        # Check if node list is not empty
        if len(node_list) > 0:
            # Get the next child
            child = node_list.pop()
    
            # Define the probability
            p = round(random.uniform(0, 1), 4)
    
            # Add nodes and update visited
            if child not in visited and label_dict[child] != 1:
                # Neighbors of child
                child_neighbors = neighbors(child)
    
                # Collect the degree of the child
                degree_child = G.degree(child)
    
                # Check if probability constraint is satisfied
                if (p <= min(1, degree_parent / degree_child) and child in list(neighbors(parent_node))):
                    # Update the edges and other list
                    edges.append((parent_node, child))

                    # Update the global list
                    set_all_added_nodes.update((parent_node, child))

                    # Uodate the visited and ratio
                    visited.add(child)
                    ratio[label_dict[child]] += 1
                    
                    # Replace the data
                    parent_node = child
                    degree_parent = degree_child
                    node_list.clear()
                    node_list.update(child_neighbors)

        else:
            # Update the nodelist with random nodes
            node_list.update(set(random.sample(list(set(G.nodes()) - set_all_added_nodes), 3)))
            
            # Get the new parent
            parent_node = node_list.pop()
            
            # Refresh the node list
            degree_parent = G.degree(parent_node)
            node_list.clear()
            node_list.update(list(neighbors(parent_node)))

    # Return the data
    return edges

def load_local_data_store(data_dir:str) -> Store:   
    # build the store
    store = Store(
        base_dir=data_dir,
        protocol='file'
    )
    return store


def fit_scaler(graph_data: Dict):
    print("Fitting Scaler...")
    scaler = MinMaxScaler()
    for p in tqdm(graph_data):
        df_f = pd.read_parquet(f"/home/kjkr7373/projects/def-pbranco/kjkr7373/BitcoinFraudDetection/data/dataset/cache/features/features_{p}.parquet")
        X = df_f[FEATURE_COLUMNS].fillna(value=0.0).values
        scaler.partial_fit(X)
        del df_f, X
        gc.collect()
    return scaler


if __name__ == "__main__":
    # Define some values
    factor = 150
    data_dir = "/home/kjkr7373/projects/def-pbranco/kjkr7373/BitcoinFraudDetection/data/dataset"
    save_path = "/home/kjkr7373/projects/def-pbranco/kjkr7373/BitcoinFraudDetection/data/datasetMHRW"
    partitions_dir = "partitions.parquet"
    splits = ['train', 'val', 'test']
    
    # Load the datastore file
    datastore = load_local_data_store(data_dir)
    df_p = pd.read_parquet(
            datastore.open_file(partitions_dir)
        ).reset_index(drop=True).reset_index()
    
    # Make the graph from it
    counters = {}
    graph_data = {}
    labelled = {}
    others = {}
    for sp, A in df_p.groupby('split'):
        graph_data[sp] = {
            x: None
            for x in sorted(A['index'])
        }
        labelled[sp] = {
            x: None
            for x in sorted(A['index'])
        }
        counters[sp] = Counter()
        
    # Fit the scalar
    scaler = fit_scaler(graph_data['train'])
    
    # New parition file
    p_dict = {'index': [], 'split': []}

    # Placeholder
    idx = 0

    # Loop over splits
    for sp in splits:

        # Loop over split data
        bar = tqdm(graph_data[sp])

        # Loop over the bar
        for p in bar:
            # Load the required files
            network = pd.read_parquet(f'{data_dir}/cache/edges/edges_{p}.parquet')
            labels = pd.read_parquet(f'{data_dir}/labels/labels_{p}.parquet')
            features = pd.read_parquet(f'{data_dir}/cache/features/features_{p}.parquet')

            # Convert to network x graph
            G = nx.from_pandas_edgelist(network, 'from', 'to')
            successors = G.neighbors # can retrieve all neighbors of a particular node with []
            
            # Replace all label 2 as label 0
            labels.loc[labels['label'] == 2, 'label'] = 0
            label_dict = dict(zip(labels.node, labels.label))
        
            # Construct graph from each positive node
            for _, pos_node in enumerate(labels[labels['label']==1].node.values):
                try: # the node in the label parquet may not exist in the edge parquet
                    result = generic_mhrw_edges(G, pos_node, label_dict, successors, factor=factor)
                except Exception as e:
                    continue
        
                # Convert the FS seach to dataframe from merging and stuff
                df_result = pd.DataFrame(result, columns=['from', 'to'])
                
                # Undirected to directed
                directed1 = network.merge(df_result, how='inner', left_on=['from', 'to'], right_on=['from', 'to'])
                directed2 = network.merge(df_result, how='inner', left_on=['from', 'to'], right_on=['to', 'from'])[['from_x', 'to_x', 'partition']].rename(columns={"from_x": "from", "to_x": "to", 'partition': 'partition'})
                samples = pd.concat([directed1, directed2], axis=0)
                edges_list = samples[['from', 'to']].values
        
                # Get the unique graph edges
                df_node = pd.DataFrame(set(edges_list.reshape(-1)), columns=['node'])
        
                # Process the features and labels
                sample_labels = labels.merge(df_node, how='inner').sort_values(by=['node'])
                sample_features = features.merge(sample_labels, how='inner', on='txid').sort_values(by=['node']).drop(['node'], axis=1)
                mapping = dict(zip(sample_labels.node.values, range(len(sample_labels))))
                samples[['from', 'to']] = samples[['from', 'to']].replace(mapping)
                sample_labels.node = [i for i in range(len(sample_labels))]
                sample_features[FEATURE_COLUMNS] = scaler.transform(sample_features[FEATURE_COLUMNS].values)
                
                try:
                    assert samples["from"].max() < 1000, "Error Occured"
                    assert samples["to"].max() < 1000, "Error Occured"
                except:
                    continue
        
                # Save the data
                samples.to_parquet(f'{save_path}/cache/edges/edges_{idx}.parquet')
                sample_labels.to_parquet(f'{save_path}/labels/labels_{idx}.parquet')
                sample_features.to_parquet(f'{save_path}/cache/features/features_{idx}.parquet')
        
                # New partition deck
                p_dict['index'].append(idx)
                p_dict['split'].append(sp)
        
                # Increment the idx
                idx += 1

                # Set the bar description
                bar.set_description(f"File name : {p} | Saved File Name : {idx}")

            # delete the extra stuff and clear memory
            del network, labels, features, G, successors
            gc.collect()

    # Save the parition file
    pd.DataFrame.from_dict(p_dict).to_parquet(f"{save_path}/partitions.parquet")