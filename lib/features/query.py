# IBM Research Singapore, 2022

import duckdb
import pyarrow.dataset as ds
import pandas as pd
from typing import Dict, Optional, Union, List, Any

from typing import Callable
import os

from lib.utils import merge_dfs

def _get_pyarrow_dataset(path: str):
    return ds.dataset(
        path,
        partitioning="hive",
        format="parquet",
    )

# ------------------------------------------------------------------
# QueryEngine
# - stores duckdb connection
# - stores pyarrow tables
# - execute 
# ------------------------------------------------------------------
class QueryEngine:

    def __init__(
        self,
        reference_dir: str,
        conn: Union[duckdb.DuckDBPyConnection,str],
        dataset_loader: Optional[Callable] = None,
        table_paths: Dict[str, str] = { # path -> name
            "vin": "vin", 
            "vout": "vout", 
            "txs": "txs",
        },
    ):
        if isinstance(conn, str):
            self.conn_database = conn
            self.conn = duckdb.connect(database=self.conn_database, read_only=True)
        else:
            self.conn_database = None # because we cant derive this from conn
            self.conn = conn
        self.reference_dir = reference_dir # where the parquets are kept

        _dataset_loader = dataset_loader
        if _dataset_loader is None:
            _dataset_loader = lambda path: _get_pyarrow_dataset(os.path.join
            (
                self.reference_dir, path
            ))
        self.tables = {}
        for path, name in table_paths.items():
            self.tables[name] = _dataset_loader(path)

    def execute(
        self, 
        statement:str, 
        order_by:str=None, 
        explain:bool=False,
        tables: List[str] = [
            "vin", "vout", "txs",
        ],
    ) -> duckdb.DuckDBPyConnection:
        for name in tables:
            locals()[name] = self.tables[name]
        if order_by:
            query_statement = f"{statement} order by {order_by} desc"
        else:
            query_statement = statement
        query_statement = f"explain {query_statement}" if explain==True else query_statement

        return self.conn.execute(query_statement)

# ------------------------------------------------------------------
# Query 
# - bind SQL statement to target_heit
# - bind SQL stateent to start_heit, end_heit
# - query via executing the engine
# ------------------------------------------------------------------
class Query:

    def __init__(
        self,
        statement: str,
        window: int = 0,
        sub_statement: str = '',
    ):
        self.window = window
        self.statement = statement
        self.sub_statement = sub_statement

    # NOTE: statement here can be provided to override
    # the original statement in the constructor
    def query(
        self, 
        target_heit:int, engine: QueryEngine, 
        statement: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:

        # if not provided we use the constructor statement
        if statement is None:
            statement = self.statement

        if self.window == 0:
            statement = statement.format(
                target_heit=target_heit,
            )
        else:
            statement = statement.format(
                target_heit=target_heit,
                start_heit=max(target_heit-self.window, 1),
                end_heit=target_heit+self.window,
            )

        conn = engine.execute(statement=statement, **kwargs)
        return conn.df()

class QuerySubStatementOne(Query):

    def __init__(
        self, 
        statement: str,
        *args, 
        sub_statement: str = '',
        **kwargs,
    ):
        super().__init__('', *args, **kwargs)
        self.sub_statement = sub_statement
        self.statement = statement
    
    def query(
        self, 
        target_heit:int, engine: QueryEngine, 
        **kwargs,
    ) -> pd.DataFrame:

        if self.window == 0:
            sub_statement = self.sub_statement.format(
                target_heit=target_heit,
            )
        else:
            sub_statement = self.sub_statement.format(
                target_heit=target_heit,
                start_heit=max(target_heit-self.window, 1),
                end_heit=target_heit+self.window,
            )

        return super().query(
            target_heit=target_heit, engine=engine,
            statement=self.statement.format(
                sub_statement=sub_statement,
            ),
            **kwargs
        )

class QueryClusterCoefficients(Query):

    def __init__(
        self,
        query_degrees: Dict[str, Query],
        query_clusters: Query,
        # clusters_type: str,
        *args,
        **kwargs,
    ):
        super().__init__('', *args, **kwargs)
        self.query_degrees = query_degrees
        self.query_clusters = query_clusters
        # self.clusters_type = clusters_type

    def query(
        self, 
        target_heit:int, engine: QueryEngine, 
        **kwargs,
    ) -> pd.DataFrame:
        neighbours_df = merge_dfs(
            [
                q.query(target_heit, engine)
                for q in self.query_degrees.values()
            ],
            on='txid',
            how='outer',
            fillna=0., 
        )

        sets = list(self.query_degrees.keys())
        if len(sets) == 2:
            cluster_name = 'in_out_cluster_coefficient'
            s1, s2 = sets[0], sets[1]
            neighbours_df["denom"] = neighbours_df[s1] * neighbours_df[s2]
        elif len(sets) == 1 :
            cluster_name = 'in_in_cluster_coefficient' if 'in_deg' in self.query_degrees.keys() else 'out_out_cluster_coefficient'
            s1 = sets[0]
            neighbours_df["denom"] = neighbours_df[s1] * (neighbours_df[s1]-1)
        else:
            raise NotImplementedError
        
        extracted_triangles = self.query_clusters.query(
            target_heit, engine,
        ).drop_duplicates().groupby("txid").size().reset_index(
        ).rename(columns={'index':'txid', 0: 'num_triangles'})

        combined_df = merge_dfs(
            [extracted_triangles, neighbours_df],
            how='outer',
            fillna=0., 
        )
        combined_df[cluster_name] = combined_df['num_triangles']/combined_df["denom"]
        return (
            combined_df[['txid', cluster_name]]
        ).fillna(0.) # the divide by zeros will be addressed by fillna
