# Set dataset paths
TRAIN_DATASET="../../data/mp20_nonmetal/train_data_reduce_zero.csv"
VAL_DATASET="../../data/mp20_nonmetal/val_data_reduce_zero.csv"
# Set column indices for slices and property
SLICES_COLUMN_INDEX=0
PROP_COLUMN_INDEX_LIST="2"  # Example for one property column

python train.py \
  --run_name bandgap \
  --batch_size 36 \
  --max_epochs 20 \
  --n_embd 512 \
  --n_layer 8 \
  --n_head 8 \
  --learning_rate 3.3e-4 \
  --train_dataset $TRAIN_DATASET \
  --val_dataset $VAL_DATASET \
  --slices_column_index $SLICES_COLUMN_INDEX \
  --prop_column_index_list $PROP_COLUMN_INDEX_LIST
