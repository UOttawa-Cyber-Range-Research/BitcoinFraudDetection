# IBM Research Singapore, 2022

from lib.features.query import Query, QuerySubStatementOne, QueryClusterCoefficients

# ------------------------------------------------------------------
# Query Definitions
# ------------------------------------------------------------------

# for each pkey that has transacted, find the difference between max heit and min heit
query_lifetime = Query(
    """
        select 
            pkey.pkey, max(pkey.heit)-min(pkey.heit)+1 as lifetime
            from pkey
            inner join
            vout
            on pkey.pkey == vout.pkey
            where           
            pkey.heit between {start_heit} and {end_heit}
            and vout.heit between {start_heit} and {end_heit}
            and vout.vheit between {start_heit} and {end_heit}
            group by pkey.pkey
    """,
    window=100
)

# count number of heits where pkey actively occurs in vout
query_active_heits = QuerySubStatementOne(
    """
        select 
            pkey.pkey, count(distinct active_txs.vheit) as num_active_heits
            from pkey
            inner join
            ({sub_statement}) as active_txs
            on pkey.pkey == active_txs.pkey
            where           
            pkey.heit between {{start_heit}} and {{end_heit}}
            group by pkey.pkey
    """,
    sub_statement= """
        select
        pkey, vheit
        from vout 
        where 
        vout.vheit between {start_heit} and {end_heit}
        and
        vout.heit between {start_heit} and {end_heit}        
    """,
    window=100,
)

#gini = ( 2 * sum(x)*sum(y) / n*sum(y) ) - (count(x) + 1)/count(x)
#regr_count = (SUM(x*y) - SUM(x) * SUM(y) / COUNT(*)) / COUNT(*)
#gini_sql = 2*regr_count(x,y)/sum(y)  
#gini_sql = 2*((SUM(x*y) - SUM(x) * SUM(y) / COUNT(*)) / COUNT(*))/SUM(y)
# 2*((SUM(x*y) - SUM(x) * SUM(y) / COUNT(*)) / COUNT(*)) / SUM(y)

query_gini_inequality = QuerySubStatementOne(
    """
        select 
            pkey, 2*covar_pop(amt,row_num)/sum(amt) as gini from ({sub_statement}) group by pkey
    """,
    sub_statement= """
        select 
            pkey.pkey, vout.amt, row_number() over (partition by pkey.pkey order by vout.amt asc) as row_num
            from
            pkey
            inner join
            vout
            on pkey.pkey == vout.pkey
            where
            vout.vheit between {start_heit} and {end_heit}
            and
            vout.heit between {start_heit} and {end_heit}
            and
            pkey.heit between {start_heit} and {end_heit}  
    """,
    window=100,
)

query_gini_inequality_check = QuerySubStatementOne(
    """
        select 
            pkey, (2*covar_pop(amt,row_num)/sum(amt)) - (2*((sum(row_num*amt) - sum(row_num) * sum(amt) / count(*)) / count(*))/sum(amt)) as gini from ({sub_statement}) group by pkey
    """,
    sub_statement= """
        select 
            pkey.pkey, vout.amt, row_number() over (partition by pkey.pkey order by vout.amt asc) as row_num
            from
            pkey
            inner join
            vout
            on pkey.pkey == vout.pkey
            where
            vout.vheit between {start_heit} and {end_heit}
            and
            pkey.heit between {start_heit} and {end_heit}  
    """,
    window=100,
)

#number of distinct addresses that transfer to/from, aggregated over each transaction in block
query_sent_distinct = Query(
    """
        select pkey, count(amt) as from_count
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
        group by pkey
    """,
    window=100,
)

#number of distinct addresses that transfer to/from, aggregated over each transaction in block
query_recv_distinct = Query(
    """
        select pkey, count(amt) as to_count
        from vout
        where
        heit = {target_heit}
        group by pkey
    """
)

query_num_sent = Query(
    """
        select vin.txid, count(pkey) as num_sent
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

query_num_recv = Query(
    """
        select txid, count(pkey) as num_recv
        from vout
        where
        heit = {target_heit}
        group by txid
    """
)

# heit delay over sent and received
# avg difference of receive -> address -> sent 
# max difference of receive -> address -> sent 

# query_transfer_max_delay = QuerySubStatementOne(
#     """
#         select 
#             pkey.pkey, max(distinct active_txs.vheit)-min(distinct active_txs.vheit) as max_delay
#             from pkey
#             inner join
#             ({sub_statement}) as active_txs
#             on pkey.pkey == active_txs.pkey
#             where           
#             pkey.heit between {{start_heit}} and {{end_heit}}
#             group by pkey.pkey
#     """,
#     sub_statement= """
#         select
#         pkey, vheit
#         from vout 
#         where 
#         vout.vheit between {start_heit} and {end_heit}
#         and
#         vout.heit between {start_heit} and {end_heit}        
#     """,
#     window=100,
# )


query_transfer_delay = QuerySubStatementOne(
    """
        select 
            pkey, max(vheit - prev_vheit) as max_delay, avg(vheit - prev_vheit) as avg_delay from ({sub_statement}) group by pkey
    """,
    sub_statement= """
        select 
            pkey, vheit, lag(vheit,1,{start_heit}) over (partition by pkey order by vheit asc) as prev_vheit
            from vout 
            where 
            vout.vheit between {start_heit} and {end_heit}
            and
            vout.heit between {start_heit} and {end_heit}
    """,
    window=100,
)
