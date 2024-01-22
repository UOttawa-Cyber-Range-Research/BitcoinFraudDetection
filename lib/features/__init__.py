# IBM Research Singapore, 2022

# generate features using pyarrow and duckdb-SQL queries

import pandas as pd
import os

import pyarrow
import pyarrow.parquet as pq

import timeit
from typing import Dict, List, Any, Optional, Union, Tuple

from lib.utils import merge_dfs
from lib.features.query import QueryEngine, Query

# Definitions from Features V1
from lib.features.definitions import query_in_degree
from lib.features.definitions import query_out_degree
from lib.features.definitions import query_transaction_amount_sent
from lib.features.definitions import query_transaction_amount_received
# from lib.definitions import query_in_out_clusters
# from lib.definitions import query_out_out_clusters
# from lib.definitions import query_in_in_clusters
from lib.features.definitions import query_reuse_counts
from lib.features.definitions import query_in_out_cluster_coef
from lib.features.definitions import query_out_out_cluster_coef
from lib.features.definitions import query_in_in_cluster_coef

# Definitions from Features V2
from lib.features.definitions2 import query_lifetime
from lib.features.definitions2 import query_active_heits
from lib.features.definitions2 import query_gini_inequality
from lib.features.definitions2 import query_sent_distinct
from lib.features.definitions2 import query_recv_distinct
from lib.features.definitions2 import query_transfer_delay


def _to_parquet(
    dataframe:pd.DataFrame, 
    dir:str,
    partition_cols:List,
    **kwargs,
) -> None:
    table = pyarrow.Table.from_pandas(dataframe, preserve_index=False)

    pq.write_to_dataset(
        table,
        root_path=dir,
        partition_cols=partition_cols if len(partition_cols)>0 else None,
        existing_data_behavior='delete_matching',
        basename_template = "features-{i}.parquet",
        **kwargs,
    )

# ------------------------------------------------------------------
# FeatureProcessor
# - one per feature partition
# - holds all the queries and engine
# ------------------------------------------------------------------

class FeatureProcessor:

    def __init__(
        self, 
        base_dir: str,
        engine: Optional[QueryEngine] = None,
        dataset_name: str = '',
        queries : Dict[str, Query] = [],
    ):
        self.engine = engine
        self.queries = queries
        self.base_dir = base_dir
        self.dataset_name = dataset_name # name of the new pyarrow dataset

    def evaluate_performance(self, target_heit: int):
        perf = pd.Series({k: timeit.Timer(lambda: q.query(
            target_heit=target_heit,
            engine=self.engine
        )).timeit(number=1) 
        for k, q in self.queries.items()}).T
        return perf

    # call to generate features
    # - behavior depends on specific implementation
    def generate_features(self, target_heit: int):
        raise NotImplementedError

    # final postprocessing
    @staticmethod
    def postprocess_features(dfs: List[pd.DataFrame], target_heit: int, merge_on:str): 
        if len(dfs) == 0:
            return pd.DataFrame()
        df = merge_dfs(
            dfs,
            on=[merge_on],
            how='outer',
            fillna=0., # FIXME: is this optimal?
        )
        df[merge_on] = df[merge_on].astype(str)
        df['heit'] = target_heit
        return df

    # call this to write a generatedf (partitioned by heit) 
    # into the table
    def to_parquet(self, df: pd.DataFrame) -> None:
        _to_parquet(
            df, 
            os.path.join(self.base_dir, self.dataset_name),
        )

class FeatureProcessorSerial(FeatureProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # in the serial case, this will generate the df directly
    def generate_features(self, target_heit: int, merge_on:str='txid') -> pd.DataFrame:
        dfs = [
            q.query(
                target_heit=target_heit,
                engine=self.engine
            ) 
            for q in self.queries.values()
        ]
        return FeatureProcessor.postprocess_features(
            dfs, target_heit=target_heit, merge_on=merge_on
        )

# implemented using multiprocessing.Process
import multiprocessing as mp

# this wrapper is to paralize the query
# - we cannot pickle connection objects, so we must open a conn per query
# - we close the connection after we are done
# - this also implies we need to pickle the Query object, so we must
#   ensure its pickalable if we choose to use FeatureProcessorParallelMP
def _wrapper_parallel_query(q: Query, target_heit, reference_dir: str, conn_database: str):
    engine = QueryEngine(reference_dir, conn=conn_database)
    df = q.query(target_heit, engine)
    engine.conn.close() # close the duckdb connection
    return df

class FeatureProcessorParallelMP(FeatureProcessor):

    def __init__(
        self, 
        base_dir: str,
        conn_database: str='',
        num_workers: int=1, 
        timeout: int = 300, # default timeout = 5 mins (300 secs)
        **kwargs,
    ):
        super().__init__(base_dir, **kwargs)
        self.reference_dir = base_dir
        self.conn_database = conn_database
        self.timeout = timeout

        self.pool = mp.Pool(processes=num_workers)
        self.results = {}

        # completed
        # if not in - not started
        # if in but False - in progress
        # if in and True - completed
        self.completed = {}

    # shutdown the pool
    def shutdown(self):
        self.pool.close()

    # check if a target heit is completed
    # - completed == no more pending
    # - if completed returns true
    # - jobs can error, in which case the error notes will store
    # - if block = True, will block until job completes or timeout
    def _check(self, target_heit: int, block: bool=False) -> bool:

        # either not yet triggered or completed before
        if target_heit not in self.results:
            return False

        if (
            self.completed[target_heit]
            and 'done' in self.results[target_heit]
        ):
            return True

        # else we need to check it
        pending = []
        errored = []
        done = []

        q: mp.pool.AsyncResult
        for q in self.results[target_heit]['pending']:

            # query_check depends on blocking or not
            try:
                if block:
                    # this may throw 
                    q.wait(self.timeout)

                # if we didnt block, then check if the result is ready
                if q.ready():
                    # two possibilities
                    # - if q.successful() == True then result will return
                    # - otherwise q.get() will raise the same exception
                    #   for which the call errored
                    done.append(q.get())
                else:
                    # in the case of non-blocking, we can have cases
                    # where we are still pending, so we come here 
                    # and save the pool.AsyncResult
                    pending.append(q)
            except Exception as e:
                # will come here if there was a timeout
                # or if q.get errored
                errored.append(str(e))

        # update
        self.results[target_heit]['done'] += done
        self.results[target_heit]['errored'] += errored
        self.results[target_heit]['pending'] = pending

        if len(pending) == 0:
            # update that it has completed
            self.completed[target_heit] = True
            return True
        return False

    # for this implementation, generate features submit 
    # the jobs but do not return here
    def generate_features(self, target_heit: int) -> None:

        # check two conditions
        # - completed flag is raised
        # - done results still exist
        if (
            self.completed.get(target_heit)
            and
            ('done' in self.results[target_heit]) 
        ):
            # dont do anything if already done
            return 

        # otherwise
        # update the status to "in progress"
        self.completed[target_heit] = False
        self.results[target_heit] = {
            'done': [],
            'errored': [],
        }

        self.results[target_heit][
            'pending'
        ] = [
            self.pool.apply_async(
                _wrapper_parallel_query,
                (q, target_heit, self.reference_dir, self.conn_database),
            )
            for q in self.queries.values()
        ]

    # for this implementation, call this after generate_features
    # to return the finally computed features
    def finalize_features(
        self, target_heit: int,
        consume: bool = True, # will delete the df from done
        return_errors: bool=False,
        merge_on:str = 'txid'

    ) -> Union[
        pd.DataFrame,
        Tuple[pd.DataFrame,List]
    ]:
        self._check(target_heit, block=True)
        # there should be no more pending
        assert(self.completed[target_heit])
        df = FeatureProcessor.postprocess_features(
            self.results[target_heit]['done'],
            target_heit=target_heit,
            merge_on=merge_on
        )
        if consume:
            # if true, we will remove the dataframe that was computed
            del self.results[target_heit]['done']

        if return_errors:
            # if true, we also return a list of errors
            return df, self.results[target_heit]['errored']
        else:
            return df

# ------------------------------------------------------------------
# FeatureProcessor Definitions
# ------------------------------------------------------------------

def evaluate_performance(target_heit: int, queries:Dict[str, Query], engine:QueryEngine):
    perf = pd.Series({k: timeit.Timer(lambda: q.query(
        target_heit=target_heit,
        engine=engine
    )).timeit(number=1) 
    for k, q in queries.items()}).T
    return perf

def features_not_windowed_one(
    cls: FeatureProcessor,
    *args,
    **kwargs,
):
    return cls(
        *args,
        dataset_name='not_windowed/txs_features_one',
        queries = {
            'in_deg': query_in_degree,
            'out_deg': query_out_degree,
            "txs_amt_recv_stats": query_transaction_amount_received,
        },
        **kwargs,
    )

def features_windowed_one(
    cls: FeatureProcessor,
    *args,
    **kwargs,
):
    return cls(
        *args,
        dataset_name='100heit_window/txs_features_one',
        queries = {
            "txs_amt_sent_stats": query_transaction_amount_sent,
            'in_out_cc': query_in_out_cluster_coef,
            'in_in_cc': query_in_in_cluster_coef,
            'out_out_cc': query_out_out_cluster_coef
        },
        **kwargs,
    )

def features_addr_one(
    cls: FeatureProcessor,
    *args,
    **kwargs,
):
    return cls(
        *args,
        dataset_name='100heit_window/adr_features_one',
        queries = {
            'addr_reuse_count': query_reuse_counts,
        },
        **kwargs,
    )

def features_addr_two(
    cls: FeatureProcessor,
    *args,
    **kwargs,
):
    return cls(
        *args,
        dataset_name='100heit_window/adr_features_two',
        queries = {
            'lifetime': query_lifetime,
            'active_heits': query_active_heits,
            'gini_inequality': query_gini_inequality,
            'transfer_delay_heits': query_transfer_delay,
            'transfer_sent_distinct': query_sent_distinct,
            'transfer_recv_distinct': query_recv_distinct,
        },
        **kwargs,
    )

# from lib.features.definitions3 import query_max_balance_difference

# def features_addr_three(
#     cls: FeatureProcessor,
#     *args,
#     **kwargs,
# ):
#     return cls(
#         *args,
#         dataset_name='100heit_window/adr_features_three',
#         queries = {
#             'max_balance_difference': query_max_balance_difference,
#         },
#         **kwargs,
#     )
