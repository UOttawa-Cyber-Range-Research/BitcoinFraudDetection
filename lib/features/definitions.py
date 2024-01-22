# IBM Research Singapore, 2022

from lib.features.query import Query, QuerySubStatementOne, QueryClusterCoefficients

# ------------------------------------------------------------------
# Query Definitions
# ------------------------------------------------------------------
query_in_degree = Query(
    """
        select 
        txid, count(vin) as in_deg
        from vin 
        where 
        heit = {target_heit}
        group by txid
    """
)

query_out_degree = Query(
    """
        select 
        txid, max(indx) + 1 as out_deg
        from vout 
        where 
        heit = {target_heit}
        group by txid
    """
)

query_transaction_amount_sent = Query(
    """
        select vin.txid, sum(amt) as total_amt_sent, mean(amt) as avg_amt_sent, stddev(amt) as sd_amt_sent
        from
        vin
        inner join
        vout
        on vin.txid=vout.vout and vin.vin = vout.txid
        where 
        vin.vheit between {start_heit} and {target_heit}
        and
        vin.heit = {target_heit}
        and
        vout.heit between {start_heit} and {target_heit}
        and
        vout.vheit = {target_heit}
        group by vin.txid
    """,
    window=100,
)

query_transaction_amount_received = Query(
    """
        select txid, sum(amt) as total_amt_recv, mean(amt) as avg_amt_recv, stddev(amt) as sd_amt_recv
        from vout
        where
        heit = {target_heit}
        group by txid
    """
)

query_reuse_counts = Query(
    '''
        select 
            pkey.pkey, sum(pkey.cnt) as reuse_count
            from pkey
            inner join
            vout
            on pkey.pkey = vout.pkey
            where
            vout.heit = {target_heit}
            and pkey.heit between {start_heit} and {end_heit}
            group by pkey.pkey
    ''',
    window=100
)


## NOTE: for QuerySubStatementOne, 
# - variables in statement are templated by DOUBLE braces
# - variables in sub_statement are templated by SINGLE brace
query_in_out_clusters = QuerySubStatementOne(
    """
        select 
        vout.txid as vin, A.txid, vout.vout
        from ({sub_statement}) as A, vout 
        where 
        vout.txid = A.vin 
        and 
        vout.vout = A.vout
        and 
        vout.heit between {{start_heit}} and {{end_heit}}
        and 
        vout.vheit between {{start_heit}} and {{end_heit}}
    """,
    sub_statement= """
        select 
        vin.vin, vin.vindx, vin.vheit as vinheit, vin.txid, vout.vout, vout.indx, vout.vheit
        from vin, vout 
        where 
        vin.heit = '{target_heit}'
        and 
        vout.heit = '{target_heit}'
        and
        vin.txid = vout.txid
    """,
    window=100,
)

query_out_out_clusters = QuerySubStatementOne(
    """
        select 
        A.txid as vout1, A.vout as vout2, A.vin as txid
        from ({sub_statement}) as A, vout 
        where 
        vout.vout = A.vout
        and 
        vout.txid = A.vin
        and
        vout.vheit = {{target_heit}}
        and
        vout.heit between {{target_heit}} and {{end_heit}}
    """,
    sub_statement= """
        select 
        vin.vin, vin.vindx, vin.vheit as vinheit, vin.txid, vout.vout, vout.indx, vout.vheit
        from vin, vout 
        where 
        vin.vheit = {target_heit}
        and 
        vout.heit between {target_heit} and {end_heit}
        and
        vout.vheit between {target_heit} and {end_heit}
        and
        vin.heit between {target_heit} and {end_heit}
        and
        vout.txid = vin.txid
    """,
    window=100,
)

query_in_in_clusters = QuerySubStatementOne(
    """
        select 
        A.vin as vin1, A.txid as vin2, A.vout as txid
        from ({sub_statement}) as A, vout 
        where 
        vout.txid = A.vin
        and
        vout.vout = A.vout
        and
        vout.vheit = {{target_heit}}
        and
        vout.heit between {{start_heit}} and {{target_heit}}
    """,
    sub_statement= """
        select 
        vin.vin, vin.vindx, vin.vheit as vinheit, vin.txid, vout.vout, vout.indx, vout.vheit
        from vin, vout 
        where 
        vout.vheit = '{target_heit}'
        and 
        vout.heit between {start_heit} and {target_heit}
        and
        vin.vheit between {start_heit} and {target_heit}
        and
        vin.heit between {start_heit} and {target_heit}
        and
        vout.txid = vin.txid
    """,
    window=100,
)

# NOTE: query_degrees should be keyed according to how
# - the degrees will be returned in the dataframes
query_in_out_cluster_coef = QueryClusterCoefficients(
    query_degrees={
        'in_deg': query_in_degree, 
        'out_deg': query_out_degree,
    },
    query_clusters=query_in_out_clusters,
)

query_in_in_cluster_coef = QueryClusterCoefficients(
    query_degrees={
        'in_deg': query_in_degree, 
    },
    query_clusters=query_in_in_clusters,
)

query_out_out_cluster_coef = QueryClusterCoefficients(
    query_degrees={
        'out_deg': query_out_degree,
    },
    query_clusters=query_out_out_clusters,
)

