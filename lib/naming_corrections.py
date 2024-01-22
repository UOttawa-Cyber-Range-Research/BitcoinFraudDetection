TABLES_V5_2_V4_RENAME = {
    "not_windowed/txs_features_one": {
        'in_deg': 'num_sent', # FIXME: because we do not have num_sent
    },
    "100heit_window/txs_features_one": {
        'in_in_cluster_coefficient': 'ave_neigh_in_in', 
        'in_out_cluster_coefficient': 'ave_neigh_in_out', 
        'out_out_cluster_coefficient': 'ave_neigh_out_out',
    }
}

TABLES_COLUMNS_DEFAULT = {
    "not_windowed/txs_features_one": [
        "in_deg", 
        "out_deg",
        "total_amt_recv",  
        "avg_amt_recv",  
        "sd_amt_recv",  
    ],
    "100heit_window/txs_features_one": [
        "total_amt_sent",  
        "avg_amt_sent",  
        "sd_amt_sent", 
        "in_in_cluster_coefficient",
        "in_out_cluster_coefficient",
        "out_out_cluster_coefficient",
    ],
    "100heit_window/adr_features_one": [
        "reuse_count",
    ],
    "100heit_window/adr_features_two": [
        "lifetime",
        "num_active_heits",
        "gini",
        "max_delay",
        "avg_delay",
        "from_count",
        "to_count",
    ],
}

### IF USING PROVIDED CACHE ###
TABLES_COLUMNS_DEFAULT_LEGACY = {
    "not_windowed/txs_features_one": [
        "in_deg", 
        "out_deg",
        "total_amt_sent",  
        "avg_amt_sent",  
        "sd_amt_sent",  
    ],
    "100heit_window/txs_features_one": [
        "total_amt_recv",  
        "avg_amt_recv",  
        "sd_amt_recv", 
        "in_in_cluster_coefficient",
        "in_out_cluster_coefficient",
        "out_out_cluster_coefficient",
    ],
    "100heit_window/adr_features_one": [
        "reuse_count",
    ],
    "100heit_window/adr_features_two": [
        "lifetime",
        "num_active_heits",
        "gini",
        "max_delay",
        "avg_delay",
        "from_count",
        "to_count",
    ],
}

### IF USING PROVIDED CACHE ###
TABLES_V5_2_V4_RENAME_LEGACY = {
    "not_windowed/txs_features_one": {
        'in_deg': 'num_sent', # FIXME: because we do not have num_sent
        'avg_amt_sent': 'ave_amt_recv',  # FIXME: only used if not regenerating cache, fixed in new features
        'total_amt_sent': 'tot_amt_recv', # FIXME: only used if not regenerating cache, fixed in new features
        'sd_amt_sent': 'sd_amt_recv',  # FIXME only used if not regenerating cache, fixed in new features
    },
    "100heit_window/txs_features_one": {
        'in_in_cluster_coefficient': 'ave_neigh_in_in', 
        'in_out_cluster_coefficient': 'ave_neigh_in_out', 
        'out_out_cluster_coefficient': 'ave_neigh_out_out',
        'avg_amt_recv': 'ave_amt_in', # FIXME: only used if not regenerating cache, fixed in new features
        'total_amt_recv': 'tot_amt_sent', # FIXME only used if not regenerating cache, fixed in new features
        'sd_amt_recv': 'sd_amt_sent', # FIXME only used if not regenerating cache, fixed in new features
    }
}

# FEATURES_NAMES_FROM_PRELOADED_CACHE = [
#     'num_sent', 
#     'total_amt_in', 
#     'avg_amt_sent', 
#     'sd_amt_sent',
#     'out_deg', 
#     'total_amt_out',
#     'avg_amt_recv', 
#     'sd_amt_recv',
# ]

FEATURES_NAMES_FROM_PRELOADED_CACHE = [
    'num_sent', 
    'ave_amt_in', 
    'tot_amt_sent', 
    'sd_amt_sent',
    'out_deg', 
    'tot_amt_recv',
    'ave_amt_out', 
    'sd_amt_recv',
]

FEATURES_NAMES_FROM_NEW_CACHE = [
    'num_sent', 
    'total_amt_sent', 
    'avg_amt_sent', 
    'sd_amt_sent',
    'out_deg', 
    'total_amt_recv',
    'avg_amt_recv', 
    'sd_amt_recv',
]

FEATURE_COLUMNS_OTHERS = [
    # 'txid', 
    # 'num_out_spent', 
    # 'partition'
    'ave_neigh_in_in', 
    'ave_neigh_in_out', 
    'ave_neigh_out_out',
    "reuse_count_mean",
    "reuse_count_std",
    "reuse_count_sum",
    "reuse_count_max",
    "reuse_count_min",
    "lifetime_mean",
    "lifetime_std",
    "lifetime_sum",
    "lifetime_max",
    "lifetime_min",
    "num_active_heits_mean",
    "num_active_heits_std",
    "num_active_heits_sum",
    "num_active_heits_max",
    "num_active_heits_min",
    "gini_mean",
    "gini_std",
    "gini_sum",
    "gini_max",
    "gini_min",
    "max_delay_mean",
    "max_delay_std",
    "max_delay_sum",
    "max_delay_max",
    "max_delay_min",
    "avg_delay_mean",
    "avg_delay_std",
    "avg_delay_sum",
    "avg_delay_max",
    "avg_delay_min",
    "from_count_mean",
    "from_count_std",
    "from_count_sum",
    "from_count_max",
    "from_count_min",
    "to_count_mean",
    "to_count_std",
    "to_count_sum",
    "to_count_max",
    "to_count_min",
]