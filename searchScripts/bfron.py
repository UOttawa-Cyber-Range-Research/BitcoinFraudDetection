import os
import gc
import pandas as pd
import numpy as np
import networkx as nx
from lib.store import Store
from tqdm import tqdm
from collections import Counter
from typing import Dict
from collections import deque
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


# Code for bfron sampling
def bfs_custom(G, source, label_dict, neighbors=None, depth_limit=None, ratio=None, visited=None):
    '''
    Generic breath first seach algorithm
    '''
    # Define the depth of the search
    if depth_limit is None:
        depth_limit = len(G)
        
    # neighbors(source): return a generator of source's neighbors
    queue = deque([(source, depth_limit, neighbors(source))])

    # Placeholder for edges
    edges = []

    # Run the queue
    while queue:
        parent, depth_now, children = queue[0]
        try:
            # Logic for random first seach
            child = next(children)

            # If the child has not been visited before
            if child not in visited:
                # Construct an edge
                edges.append((parent, child))

                # Update visited
                visited.add(child)

                # Increment the ratio
                ratio[label_dict[child]] += 1

                # Incremen the child
                if depth_now > 1:
                    queue.append((child, depth_now - 1, neighbors(child)))
                    
            # Break the ratio of the model
            try:
                if ratio[0]/ratio[1] >= factor:
                    break
            except ZeroDivisionError as e:
                pass

        # If no more nodes in the queue
        except StopIteration:
            queue.popleft()

    # Return the edges
    return edges, ratio, visited

def n_depth_search(G, frontiers, neighbors, label_dict, depth, ratio, visited, edges):
    '''
    Define the depth until the nodes are important
    '''
    # Get the nodes for the source
    new_set = set()
    for front_node in frontiers:
        # Add the front node to the visited
        visited.add(front_node)

        # Maintain the ratio
        ratio[label_dict[front_node]] += 1
        
        # Collect the nodes
        neighbors_ = neighbors(front_node)

        # Loop the neighbors
        for neigh in neighbors_:
            # Updated the visited
            visited.add(neigh)

            # Updates the edges
            edges.append([front_node, neigh])

            # Make the new frontier list
            new_set.add(neigh)

            # Add the ratio to the list
            ratio[label_dict[neigh]] += 1

    return new_set, ratio, visited, edges

def frontier_sampling(frontiers, edges, neighbors, ratio, visited):
    '''
    Performs frontier sampling on the frontier to chose initial nodes
    '''
    # Define the prob of selection 
    degrees = [G.degree(i) for i in frontiers]
    probs = [G.degree(i) / sum(degrees) for i in frontiers]

    # Choose a frontier
    chosen_frontier = np.random.choice(list(frontiers), p=probs, size=1)[0]

    # Get the neighbors of the chosen frontier
    frontier_neighbors = list(neighbors(chosen_frontier))

    # Randomly select a child node
    random_child = random.choice(frontier_neighbors)

    if random_child not in visited:
        # Add the edge
        edges.append([chosen_frontier, random_child])
    
        # Update the ratio
        ratio[label_dict[random_child]] += 1
    
        # Update the frontiers
        frontiers.remove(chosen_frontier)
        visited.add(chosen_frontier)
        frontiers.add(random_child)

    # Return frontiers
    return frontiers, edges, ratio, visited

def frontier_bfs(G, source, label_dict, neighbors=None, depth_limit=None, sort_neighbors=None, factor=100, depth=1):
    '''
    Uses frontiers to start and then uses breath first seach to complete the models
    '''
    # Define the ratio
    ratio = {0: 0, 1: 0}

    # Define the visited nodes
    visited = set()

    # Placeholder for edges
    edges = []

    # Define the frontier nodes
    frontiers = set([source])

    # Stage 1 frontiers
    for _ in range(0, depth):
        frontiers, ratio, visited, edges = n_depth_search(G, frontiers, neighbors,
                                                         label_dict, 1, ratio,
                                                         visited, edges)

    # Stage 2 sampling (fronier sampling)
    depth_val = random.randint(2, 5)
    for _ in range(depth_val):
        frontiers, edges, ratio, visited = frontier_sampling(frontiers, edges, neighbors, ratio, visited)

    assert source not in frontiers, "Error source in frontier after stage 2"

    # Update the visisted
    visited.update(frontiers)

    # Define random probs for the frontiers
    frontier_probs = list(np.random.rand(len(frontiers)))
    frontier_list = list(frontiers)

    # Sort the list based on the probs
    frontiers_with_probs_sorted = sorted([[frontier_list[i], frontier_probs[i]] for i in range(len(frontier_list))],
                                         key=lambda x : x[-1],
                                         reverse=True)
    
    # Loop over the frontiers
    for front_node, _ in frontiers_with_probs_sorted:
        # Run BFS on the frontier to collect its node
        edges_bfs, ratio, visited = bfs_custom(G=G, source=front_node, label_dict=label_dict,
                                                neighbors=neighbors, depth_limit=None,
                                                ratio=ratio, visited=visited)

        # Add to the edges
        edges.extend(edges_bfs)
    
    # Return the edges
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
    save_path = "/home/kjkr7373/projects/def-pbranco/kjkr7373/BitcoinFraudDetection/data/datasetBFON"
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
                    result = frontier_bfs(G, pos_node, label_dict, successors, factor=factor)
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