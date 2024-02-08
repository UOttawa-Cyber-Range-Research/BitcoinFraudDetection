import os
import gc
import pandas as pd
import numpy as np
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


# function for frontier
def generic_frontier_edges(G, source, label_dict, neighbors=None, depth_limit=None,
                           sort_neighbors=None, factor=100, num_frontiers=20):
    """
    Iterate over edges in a MHR search.
    """

    # Define the ratio 
    ratio = {0: 0, 1: 0}
    visited = {source}
    break_condition = False

    # Define the depth limit
    if depth_limit is None:
        depth_limit = len(G)

    # Define the frontier nodes
    frontier_nodes = set()
    neighbors_s_node = set(neighbors(source))

    # Find the reachable frontier nodes
    reachable_nodes = list(nx.single_source_shortest_path_length(G, source).keys())
    reachable_nodes.remove(source)

    # Add some random nodes just in case
    temp_ = list(np.random.choice(list(G.nodes), size=num_frontiers))
    reachable_nodes.extend(temp_)
    reachable_nodes = [i for i in reachable_nodes if i in label_dict][:num_frontiers]

    # Update the frontier nodes
    frontier_nodes.update(reachable_nodes)
    
    # Maintain the ratio
    ratio[label_dict[source]] += 1

    # Placeholder for collected edges
    edges = [(source, i) for i in frontier_nodes]

    # Add the seed nodes to the mix
    for node in frontier_nodes:
        ratio[label_dict[node]] += 1

    # Define the patience
    patience = 0

    # Loop and collect edgeneric_frontier_edges
    while (ratio[0]/ratio[1]) < factor:
        # Select a new node with some probability
        degrees = [G.degree(i) for i in frontier_nodes]
        list_probs = [G.degree(i) / sum(degrees) for i in frontier_nodes]
        
        # Get the next child
        chosen_frontier = np.random.choice(list(frontier_nodes), p=list_probs, size=1)[0]

        # Neighbors of child
        frontier_neighbors = list(neighbors(chosen_frontier))

        # Randomly chose one of them
        random_child = random.choice(frontier_neighbors)

        # Add nodes and update visited
        if random_child not in visited and label_dict[random_child] != 1:
            # Update the edges and other list
            edges.append((chosen_frontier, random_child))
            ratio[label_dict[random_child]] += 1

            # Replace u by v in the node list
            frontier_nodes.remove(chosen_frontier)
            frontier_nodes.add(random_child)

            # Reset patience
            patience = 0
        else:
            # Increment the patience
            patience += 1

            if break_condition:
                break
            
            # Edge case for unreacheable nodes
            if patience == 50:
                frontier_nodes = visited - set([source])
                break_condition = True
            
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
    save_path = "/home/kjkr7373/projects/def-pbranco/kjkr7373/BitcoinFraudDetection/data/datasetFS"
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
                    result = generic_frontier_edges(G, pos_node, label_dict, successors, factor=factor)
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