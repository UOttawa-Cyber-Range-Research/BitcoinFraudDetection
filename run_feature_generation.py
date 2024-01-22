# IBM Research Singapore, 2022

from lib.features import features_not_windowed_one, features_windowed_one, features_addr_one, features_addr_two
from lib.features import FeatureProcessorParallelMP
from lib.store import Store

from typing import List, Any

from tqdm import tqdm
import pandas as pd
import numpy as np
import os

FEATURES = {
    'features_not_windowed_one': features_not_windowed_one,
    'features_windowed_one': features_windowed_one,
    'features_addr_one': features_addr_one,
    'features_addr_two': features_addr_two
}

ERROR_FILE='run_feature_generation_errors.json'

import argparse

parser = argparse.ArgumentParser(
                    prog = 'FeatureGenerator',
                    description = 'Generates Graph features from DuckDB TxG to Feature Matrices in Parquet')

parser.add_argument('-nw', '--num_workers', type=int, default=8,
                    help='number of workers to initialize in multiprocessing pool')
parser.add_argument('-path', '--db_base_path', default="data/tables",
                    help='specify database path to get TxG from')
parser.add_argument('-path_bucket', '--bucket_base_path', default=None, # "graphs/_v5" for remote store, None for local store
                    help='specify path to store data in buckets')
parser.add_argument('-start', '--start_heit', type=int, default=619582,
                    help='start range of block heit to query from')
parser.add_argument('-end', '--end_heit', type=int, default=642430+1,
                    help='end range of block heit to query to')
parser.add_argument('-bs', '--batch_size', type=int, default=10,
                    help='batch size to group data for processing')
parser.add_argument('-feat', '--feature_list', nargs='+', default=[
    'features_not_windowed_one', 
    'features_windowed_one', 
    'features_addr_one', 
    'features_addr_two'
    ])
parser.add_argument(
    '-dry-run', action='store_true', default=False,
    help='compute but do not save processed features'
) 

# return batches
def make_batches(
    data: List[Any], 
    bs: int = 100,
    tqdm=None,
):

    n = np.ceil(len(data)/bs)
    n = int(n)

    if tqdm is None:
        R = range(n)
    else:
        R = tqdm(range(n), total=n)

    for i in R:
        yield data[bs*i:bs*(i+1)]


def initialize_feature_classes_on_multiprocessor(args: argparse.ArgumentParser):
    # for key, F, num_workers in [
    #     ("nw", features_not_windowed_one, args.num_workers),
    #     ("w", features_windowed_one, args.num_workers),
    #     ("a", features_addr_one, args.num_workers),
    # ]:

    assert len(args.feature_list)>0, "feature list cannot be empty"

    features = {}

    for key, F, num_workers in [(feat_name, FEATURES[feat_name], args.num_workers) for feat_name in args.feature_list]:
        features[key] = F(
            FeatureProcessorParallelMP,
            base_dir=args.db_base_path, 
            conn_database=os.path.join(args.db_base_path, 'database.db'),
            num_workers=num_workers,
        )
    return features

# set envvars prefixed by this

def load_local_store(path:str):
    return Store(
        base_dir=path,
        protocol = 'file'
    )


from collections import defaultdict

def main(args: argparse.ArgumentParser):
    features = initialize_feature_classes_on_multiprocessor(args)
    data_store = load_local_store(args.db_base_path)

    errors = defaultdict(dict)
    for batch in make_batches(
        range(args.start_heit, args.end_heit),
        bs=args.batch_size, # batch_size
        tqdm=tqdm,
    ):
        # go through the batch
        dfs_batch = {
            k: list() for k in features
        }

        for h in batch:
            # submit multiple heits asynchronously
            for feat in features.values():
                feat.generate_features(target_heit=h)
        
        for h in batch:
            # finalize for each heit
            for k, feat in features.items():   
                df, errs = feat.finalize_features(
                    target_heit=h, 
                    merge_on="pkey" if 'addr' in k else "txid",
                    return_errors=True,
                )

                if len(errs) > 0:
                    errors[h][k] = errs
                    print (f'heit={h}, feature={k} has {len(errors[h][k])} errors')
                else:
                    dfs_batch[k].append(df)

        # after the batch is done, we can write the parquet
        for k, feat in features.items():
            if args.dry_run == False:
                data_store.to_parquet(
                    pd.concat(dfs_batch[k]),
                    feat.dataset_name,
                )

    # close the pool when you are done
    for feat in features.values():
        feat.shutdown()

    _num_errors = len(errors)
    if _num_errors > 0:
        import json
        print (f"There are \'{_num_errors}\' errors")
        print (f"Errors printed in file: \'{ERROR_FILE}\'")
        with open(ERROR_FILE, 'w') as f:
            json.dump(errors, f, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)