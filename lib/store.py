# IBM Research Singapore, 2022

import pyarrow.dataset as ds
import pyarrow as pa

from typing import List, Union, Optional


import pandas as pd

import os

from lib.features import _to_parquet

class Store:
    """Store handles connections to the data. 
    Implemented on top of pyarrow dataset interfaces.
    """

    def __init__(self, base_dir: str, protocol:str='file', **kwargs):

        self.base_dir = base_dir
        self.protocol = protocol
        self.kwargs = kwargs
        self.init()

    def init(self):
        import fsspec

        if self.protocol == "file":
            self.fs = fsspec.filesystem(
                'file',
                default_cache_type="none",
                default_fill_cache=False,
            )
        else:
            raise NotImplementedError
        
    def list_parquets(self, path: str, tqdm=lambda x:x):
        """list_parquets returns a list of all parquets under path"""

        parquets = []
        for root, _, files in tqdm(self.fs.walk(
            os.path.join(self.base_dir, path)
        )):
            for x in files:
                if x.endswith('.parquet'):
                    parquets.append(os.path.join(root, x))

        return parquets
    
    def list_dir(self, path:str):
        filepath = os.path.join(self.base_dir, path)
        return self.fs.ls(filepath)

    def exists(self, path:str):
        filepath = os.path.join(self.base_dir, path)
        return self.fs.exists(filepath)

    def open_file(self, path:str):
        filepath = os.path.join(self.base_dir, path)
        assert self.fs.isfile(filepath), f"Path specified '{filepath}' does not point to an existing file"
        return self.fs.open(filepath)

    def remove_path(self, path:str, recursive=True):
        filepath = os.path.join(self.base_dir, path)
        self.fs.rm(filepath, recursive=recursive)

    def mkdir(self, path:str):
        filepath = os.path.join(self.base_dir, path)
        self.fs.mkdir(filepath)

    def list_parquets_range(
        self, 
        path: str, 
        start: int,
        end: Optional[int] = None,
        key: str = 'heit',
        tqdm=lambda x:x,
    ):
        """list_parquest_range list all parquets under path but partition range"""
        if end is None:
            end = start + 1

        parquets = []
        for x in range(start, end):
            for root, _, files in tqdm(self.fs.walk(
                os.path.join(
                    self.base_dir, 
                    path,
                    f"{key}={x}"
                )
            )):
                for x in files:
                    if x.endswith('.parquet'):
                        parquets.append(os.path.join(root, x))

        return parquets

    def dataset(
        self, 
        parquet_files: Union[str, List[str]],
    ):
        """dataset returns parquet.dataset given a either 
        a list of parquet_files or a path"""

        if isinstance(parquet_files, str):
            parquet_files = self.list_parquets(parquet_files)

        return ds.dataset(
            parquet_files,
            filesystem=self.fs,
            partitioning=ds.partitioning(
                pa.schema([("heit", pa.int32())]),
                flavor='hive',
            ),
            format="parquet",
        )

    
    def to_parquet(self, df: pd.DataFrame, path: str, partition:List=['heit']):
        """to_parquet writes dataframe to path"""
        _to_parquet(
            df,
            os.path.join(self.base_dir, path),
            partition_cols=partition,
            filesystem=self.fs,
        )


    def to_features_pandas(self, df: pd.DataFrame, path: str):
        """to_parquet_pandas writes dataframe to path using pandas engine"""
        # s3 cant do real directories
        dir_name = os.path.dirname(os.path.join(self.base_dir, path))
        if os.path.exists(dir_name)==False:
            self.fs.mkdir(dir_name)
        with self.fs.open(os.path.join(self.base_dir, path), "wb") as f:
            df.to_parquet(f, index=False)
