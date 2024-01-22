export DATA_TABLES=data/tables
export DATA_SET=data/dataset
export MODEL=gat

mkdir -p $DATA_TABLES
mkdir -p $DATA_SET/cache
tar -xvzf $DATA_SET/edges.tar.gz -C $DATA_SET/cache &
tar -xvzf $DATA_SET/features.tar.gz -C $DATA_SET/cache & 
tar -xvzf $DATA_SET/labels.tar.gz -C $DATA_SET/ &

tar -xvzf $DATA_TABLES/txs.tar.gz -C $DATA_TABLES/ &
tar -xvzf $DATA_TABLES/vout.tar.gz -C $DATA_TABLES/ &
tar -xvzf $DATA_TABLES/vin.tar.gz -C $DATA_TABLES/ &

echo "Data Prep Complete"