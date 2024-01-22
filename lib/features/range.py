# IBM Research Singapore, 2022

# this is a single table parquet range fetcher
# - fetched using a range query (QueryEngine)

from lib.store import Store
from lib.features.query import QueryEngine
import os
import duckdb
from typing import Optional, List, Dict

from dotenv import load_dotenv
load_dotenv()

def fetch_parquet_range(
    query_tables: List[Dict[str, str]],
    start: int, end: int,
    reference_dir: str, 
    filter_columns : Optional[List] = None,
    conn: duckdb.DuckDBPyConnection = duckdb.connect(),
    override_statement: Optional[str] = None,
    store: Store = None,
):
    # the fssspec inside Store seems to have a global
    # cache, so it doesnt matter if I create the store 
    # object for each function call

    assert store!=None, "No datastore object specified to query from"
    assert len(query_tables)>0, "Auxiliary table parameter cannot be empty"

    # Initialize main query tables
    parquets = {}
    table_paths = {}
    tables=[]
    for item in query_tables:
        table_path = item["table_path"]
        table_name = item["table_name"] 
        table_paths[table_path] = table_name
        tables.append(table_name)
        parquets[table_path] = store.list_parquets_range(table_path, start, end+1)

    engine = QueryEngine(
        reference_dir=reference_dir,
        conn=conn,
        table_paths=table_paths,
        dataset_loader=lambda name: store.dataset(parquets[name]),
    )

    if filter_columns is None:
        columns = '*'
    else:
        columns = ','.join(filter_columns)

    # the default statement takes end to be inclusive
    if override_statement is None:
        statement = """select {columns} from 
        {table_name} 
        where 
        heit between {start} and {end}
        """
    else:
        statement = override_statement

    return engine.execute(
        statement.format(
            columns=columns,
            table_name=",".join(tables),
            start=start,
            end=end,
        ),
        tables=tables
    ).df()
