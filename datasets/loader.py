# IBM Research Singapore, 2022

# loader that reads from parquets using duckdb SQL style

import pandas as pd
import os, shutil
import pyarrow.dataset as ds
import pyarrow as pa

from typing import List, Union, Optional, Dict, Callable, Tuple

# --------------------------------------------------------------
# DEFINITIONS
# --------------------------------------------------------------

# feature columns
# NOTE: to be expanded
from lib.naming_corrections import TABLES_V5_2_V4_RENAME_LEGACY, TABLES_COLUMNS_DEFAULT_LEGACY, FEATURES_NAMES_FROM_NEW_CACHE, FEATURES_NAMES_FROM_PRELOADED_CACHE, FEATURE_COLUMNS_OTHERS

FEATURE_COLUMNS = FEATURES_NAMES_FROM_PRELOADED_CACHE+FEATURE_COLUMNS_OTHERS
TABLES_V5_2_V4_RENAME = TABLES_V5_2_V4_RENAME_LEGACY
TABLES_COLUMNS_DEFAULT = TABLES_COLUMNS_DEFAULT_LEGACY

# tables that we are using for features
# - this is used by create_query_engine
TABLE_PATHS_DEFAULT = [
    "vout",
    "not_windowed/txs_features_one",
    "100heit_window/txs_features_one",
    "100heit_window/adr_features_one",
]

# the table names should be mapped using the configured
# table_paths in QueryEngine. 
#
# For example the following configuration
# e.g., "not_windowed/txs_features_one": "nw_tx1",
#       "100heit_window/txs_features_one": "onedw_tx1",
#       "100heit_window/adr_features_one": "onedw_adr1",
#
# --------------------------------------------------------------
# HELPERS (for preparing QueryEngine)
# --------------------------------------------------------------
from lib.store import Store
from lib.features import QueryEngine
from tqdm import tqdm

# scan parquet files for each table in parallel 
def scan_and_populate_parquet_lists(
    store: Store,
    tables: List[str],
    workers :int = 5,
    tqdm=tqdm,
):
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    fn = partial(store.list_parquets, tqdm=tqdm)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(fn, tables, timeout=1800)
        
    return {k:v for k,v in zip(tables, results)}

# create Query Engine
# - takes in the configuration for table paths
import duckdb

def load_local_data_store(data_dir:str) -> Store:   
    # build the store
    store = Store(
        base_dir=data_dir,
        protocol='file'
    )
    return store

# --------------------------------------------------------------
# HELPERS (for LOADERS)
# --------------------------------------------------------------

from lib.features.range import fetch_parquet_range

# - df_txid2row is a map of txid to row number
def _query_txG_from_parquets(
    reference_dir: str,
    df_txid2row: pd.DataFrame, # two columns, index, txid
    start: int,
    end : int,
    store: Store
):

    def _map(df: pd.DataFrame, old_name: str, new_name: str):

        df2 = df.merge(
            df_txid2row, # map
            left_on=old_name,
            right_on='txid',
            how='inner'
        ).rename(
            columns={'index': new_name}
        ).drop(old_name, axis=1)
        
        try:
            return df2.drop('txid', axis=1)
        except:
            return df2

    G = fetch_parquet_range(
        query_tables=[
            {"table_path":"vout", "table_name": "vout"},
        ],
        start=start, end=end,
        reference_dir=reference_dir,
        override_statement="""select txid,vout from vout 
        where 
        type!='nulldata'
        and heit between {start} and {end}
        and vheit between {start} and {end}
        """,
        store=store
    ).drop_duplicates()

    G = _map(G, old_name='txid', new_name='from')
    G = _map(G, old_name='vout', new_name='to')

    return G

def multi_thread_helper(
    function: Callable, 
    arguments: List, 
    num_workers:int=8, 
    conn: duckdb.DuckDBPyConnection=duckdb.connect(),
):
    # FIXME: not sure if this is ideal but address later
    conn.execute(f"PRAGMA threads={num_workers}")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(num_workers) as executor:
        futures = [executor.submit(function, arg) for arg in arguments]
    return [future.result() for future in as_completed(futures)]

# To extend
# - tables would be used to select specific columns
# - renames would be used to rename to avoid conflicts
def _query_features_from_parquets(
    reference_dir: str,
    df_txid2row: pd.DataFrame, # two columns, index, txid
    start: int,
    end : int,
    tables: Dict[str, List] = TABLES_COLUMNS_DEFAULT, # table -> col name
    renames: Dict[str, Dict] = TABLES_V5_2_V4_RENAME, # table -> dict of renames
    order: Optional[List[str]] = None, # to specify the final column order
    conn: duckdb.DuckDBPyConnection = duckdb.connect(),
    store: Store = None,
    num_workers: int = 1,
):
    from lib.utils import merge_dfs

    def f(table_cols: Tuple):
        table, cols = table_cols
        if 'adr_features' in table:
            # merge pkey with txid from vout
            query_statement = f'''
                select 
                txid, {", ".join([f"avg(A.{col}) as {col}_mean, coalesce(stddev(A.{col}),0) as {col}_std, sum(A.{col}) as {col}_sum, max(A.{col}) as {col}_max, min(A.{col}) as {col}_min" for col in cols])}
                from 
                vout 
                inner join 
                A
                on 
                vout.pkey==A.pkey 
                and 
                vout.heit==A.heit
                where 
                A.heit between {{start}} and {{end}}
                and
                vout.heit between {{start}} and {{end}}
                group by txid            
                '''

            df = fetch_parquet_range(
                query_tables=[
                    {"table_path":table, "table_name": "A"},
                    {"table_path":"vout", "table_name": "vout"},
                ],
                start=start, end=end,
                reference_dir=reference_dir,
                override_statement=query_statement,
                conn = conn.cursor(),
                store=store,
            )
        else:
            df = fetch_parquet_range(
                query_tables=[
                    {"table_path":table, "table_name": "A"},
                ],
                start=start, end=end,
                reference_dir=reference_dir,
                filter_columns = ['txid'] + cols,
                conn = conn.cursor(),
                store=store,
            )

        if table in renames:
            ren = renames[table]
            df = df.rename(columns=ren)

        return df

    arguments = [(table, cols) for table, cols in tables.items()]
    dfs = multi_thread_helper(f, arguments, num_workers=num_workers, conn=conn)

    df = merge_dfs(
        dfs,
        on=['txid'],
        how='outer',
        fillna=0.,
    )

    df = df_txid2row.merge(
        df, on=['txid'], 
        how='left', 
    ).fillna(0.).drop('index', axis=1)

    if order is not None:
        return df[['txid'] + order]
    return df


# --------------------------------------------------------------
# LOADERS
# --------------------------------------------------------------

# main entry point
def load_data(*args, **kwargs):

    return _load_data(
        *args, 
        **{
            k:v for k,v in kwargs.items() 
            if k in [
                'feature_column_names', 
                'debug',
                'semi_supervised',
                'semi_supervised_resample_negs',
                'semi_supervised_resample_factor', 
                'splits',
                'scaler',
                'rng',
                'features_dir',
                'refresh_cache',
            ]
        }
    )

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import List, Optional
import duckdb

from datasets.types import DATA_LABEL_TRAIN, DATA_LABEL_VAL, DATA_LABEL_TEST

import numpy as np
import torch
from torch_geometric.data import Data

def reset_cache(cached_features_dir:str, cached_edges_dir:str, datastore:Store):
    if datastore.exists(cached_features_dir):
        datastore.remove_path(cached_features_dir, recursive=True)
    if datastore.exists(cached_edges_dir):
        datastore.remove_path(cached_edges_dir, recursive=True)
    datastore.mkdir(cached_features_dir)   
    datastore.mkdir(cached_edges_dir)   

def is_empty(path:str, datastore:Store): 
    if datastore.exists(path):
        # Checking if the directory is empty or not
        if len(datastore.list_dir(path))>0:
            return False
        else:
            return True
    else:
        return False

def fit_scaler(graph_data:Dict, cached_features_dir:str, datastore)->StandardScaler:
    print ("Fitting Scaler...")
    scaler = StandardScaler()
    for p in tqdm(graph_data, disable=True):
        df_f = pd.read_parquet(
                    datastore.open_file(os.path.join(cached_features_dir, f"features_{p}.parquet"))
                )
            
        X = df_f[
            FEATURE_COLUMNS
        ].fillna(value=0.).values
        scaler.partial_fit(X)
    return scaler

def augment_labels(y:pd.DataFrame, rng:np.random.default_rng, semi_supervised:bool, semi_supervised_resample_negs:str=None, semi_supervised_resample_factor:int=None)->pd.DataFrame:
    if semi_supervised == False:
        # convert to replace class 2 with 0
        _idx, = np.where(y == 2)
        y[_idx] = 0
    elif semi_supervised_resample_negs is None:
        # dont do anything
        pass
    elif (
        (semi_supervised_resample_negs == 'random')
        or
        (semi_supervised_resample_negs == 'candidates')
    ):
        if semi_supervised_resample_negs == 'candidates':
            raise NotImplementedError("neg-candidates not implemented for loader_v3")
        else:
            # randomize the 0 and 2 labels
            _idx, = np.where((y == 2) | (y==0))
            y[_idx] = 2 # unsup

        _n = max((y==1).sum(), 1) # at least 1

        for i in rng.choice(
            range(len(_idx)),
            size=min(
                len(_idx), 
                _n * semi_supervised_resample_factor
            ), 
            replace=False,
        ):
            y[_idx[i]] = 0 # neg class

    return y

def read_from_cache(features_partition_filepath:str, edges_partition_filepath:str, datastore:Store)->Tuple[pd.DataFrame,pd.DataFrame]:
    df_f = pd.read_parquet(
        datastore.open_file(features_partition_filepath)
    )
    df_e = pd.read_parquet(
        datastore.open_file(edges_partition_filepath)
    )
    return df_f, df_e

def generate_from_queries(df_l: pd.DataFrame, _partition: pd.DataFrame, feature_datastore:Store)->Tuple[pd.DataFrame,pd.DataFrame]:
    df_e = _query_txG_from_parquets(
        feature_datastore.base_dir, 
        df_l['txid'].reset_index(), # map
        start=_partition['start'],
        end=_partition['end'],
        store=feature_datastore
    )

    df_f = _query_features_from_parquets(
        feature_datastore.base_dir, 
        df_l['txid'].reset_index(), # map
        start=_partition['start'],
        end=_partition['end'],
        order=FEATURE_COLUMNS,
        store=feature_datastore
    )

    return df_f, df_e

def _load_data(
    data_dir: str,
    device,
    feature_column_names: List[str] = FEATURE_COLUMNS,
    debug=False,
    semi_supervised=True,
    semi_supervised_resample_negs=None, # randomize negs
    semi_supervised_resample_factor=None, # randomize negs factor
    splits: List[str] = [
        DATA_LABEL_TRAIN,
        DATA_LABEL_VAL,
        DATA_LABEL_TEST,
    ], 
    scaler: Optional[StandardScaler] = None,
    rng = np.random.default_rng(seed=1),
    features_dir: Optional[str]=None, # extra path that we use for duckdb queries
    refresh_cache:bool = False,
):
    labels_dir = "labels"
    partitions_dir = "partitions.parquet"
    cached_features_dir = "cache/features"
    cached_edges_dir = "cache/edges"

    assert data_dir!=None, ("data path does not exist")
    assert features_dir!=None, ("duckdb path does not exist")

    datastore = load_local_data_store(data_dir)
    features_datastore = load_local_data_store(features_dir)

    assert datastore.exists(partitions_dir), "partitions.parquet missing"
    assert datastore.exists(labels_dir) or is_empty("labels", datastore)==False, "labels missing"

    print ("Building dataset")

    # If not using cache, clear cache folder
    global FEATURE_COLUMNS, TABLES_V5_2_V4_RENAME, TABLES_COLUMNS_DEFAULT
    if refresh_cache:
        reset_cache(cached_features_dir, cached_edges_dir, datastore)
        FEATURE_COLUMNS = FEATURES_NAMES_FROM_NEW_CACHE+FEATURE_COLUMNS_OTHERS
        from lib.naming_corrections import TABLES_V5_2_V4_RENAME, TABLES_COLUMNS_DEFAULT
        TABLES_V5_2_V4_RENAME = TABLES_V5_2_V4_RENAME
        TABLES_COLUMNS_DEFAULT = TABLES_COLUMNS_DEFAULT
        print ("Using DuckDB to generate features to cache")
    else:
        print ("Using existing cache. Verifying...")

    # read the top level partition parquet
    df_p = pd.read_parquet(
        datastore.open_file(partitions_dir)
    ).reset_index(drop=True).reset_index()

    if debug:
        df_p = df_p.groupby('split').first()

    # we make a copy of the y in its native int
    # - because we need to int version for the metrics (since we want to compute in GPU)
    # - unfortunately the loss function we are using reqiures the y values
    # to be in float
    def build_data(X, y, edges):
        return Data(
            x=torch.tensor(
                X.astype(np.float32), 
                device=device
            ), 
            edge_index=torch.tensor(edges.T, device=device), 
            y=torch.tensor(y, device=device).float(),
            y_i=torch.tensor(y, device=device)
        )

    from collections import Counter
    counters = {}
    graph_data = {}
    labelled = {}
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

    # Retrieve features and edges dataframes
    for sp in splits:
        for p in tqdm(graph_data[sp], disable=True):
            labels_partition_filepath = os.path.join(labels_dir, f"labels_{p}.parquet")
            features_partition_filepath = os.path.join(cached_features_dir, f"features_{p}.parquet")
            edges_partition_filepath = os.path.join(cached_edges_dir, f"edges_{p}.parquet")

            # Regenerate from duckdb if conditions exist
            if (
                refresh_cache==True
                or datastore.exists(features_partition_filepath)==False 
                or datastore.exists(edges_partition_filepath)==False
            ):                
                # it should hit only one entry
                _partition = df_p.query(f'index == {p}').iloc[0]

                # df_l :
                # - txid (in same order as df_f)
                # - label
                # - node (may not really be needed, not sure why another ordering )
                df_l = pd.read_parquet(
                    datastore.open_file(labels_partition_filepath)
                )

                df_f, df_e = generate_from_queries(df_l, _partition, features_datastore)

                # Stores generated queries to cache
                datastore.to_features_pandas(df_f, features_partition_filepath)
                datastore.to_features_pandas(df_e, edges_partition_filepath)

    # If it's a train loop and there is no pre-fitted scaler
    if (
        (DATA_LABEL_TRAIN in splits)
        and 
        (scaler is None)
    ):
        scaler = fit_scaler(graph_data[DATA_LABEL_TRAIN], cached_features_dir=cached_features_dir, datastore=datastore)            

    # Convert cache into a geometric dataset
    print("Loading Cached Data for Training...")
    for sp in splits:
        for p in tqdm(graph_data[sp], disable=True):
            labels_partition_filepath = os.path.join(labels_dir, f"labels_{p}.parquet")
            features_partition_filepath = os.path.join(cached_features_dir, f"features_{p}.parquet")
            edges_partition_filepath = os.path.join(cached_edges_dir, f"edges_{p}.parquet")

            # df_l :
            # - txid (in same order as df_f)
            # - label
            # - node (may not really be needed, not sure why another ordering )
            df_l = pd.read_parquet(
                datastore.open_file(labels_partition_filepath)
            )

            assert datastore.exists(features_partition_filepath), f"Missing Features Dataframe, '{features_partition_filepath}'"                
            assert datastore.exists(edges_partition_filepath), f"Missing Edges Dataframe, '{edges_partition_filepath}'"                
            df_f, df_e = read_from_cache(features_partition_filepath, edges_partition_filepath, datastore)

            X = df_f[
                feature_column_names
            ].fillna(value=0.).values

            if scaler:    
                X = scaler.transform(X)

            # need to ensure ordering is same
            y = df_f[['txid']].merge(
                df_l[['txid', 'label']],
            )['label'].values

            # uses a negative sampling strategy
            y=augment_labels(
                y, 
                rng, 
                semi_supervised=semi_supervised, 
                semi_supervised_resample_negs=semi_supervised_resample_negs, 
                semi_supervised_resample_factor=semi_supervised_resample_factor
                )

            # if semi sup is disabled, then this will essentially
            # be all of them
            labelled[sp][p], = np.where(
                y != 2
            )

            counters[sp].update(y)
            graph_data[sp][p] = build_data(
                X=X,
                y=y,
                edges=df_e[['from', 'to']].values
            )

    for sp in splits:
        print (sp, counters[sp])

    return graph_data, labelled, scaler, feature_column_names