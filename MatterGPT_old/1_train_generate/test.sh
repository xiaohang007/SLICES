# Set dataset paths
TRAIN_DATASET="../0_dataset/train_data.csv"
VAL_DATASET="../0_dataset/val_data.csv"
# Set column indices for slices and property
SLICES_COLUMN_INDEX=0
PROP_COLUMN_INDEX_LIST="1 2"  # Example for one property column

python train.py \
  --run_name eform \
  --batch_size 36 \
  --max_epochs 10 \
  --n_embd 256 \
  --n_layer 4 \
  --n_head 4 \
  --learning_rate 3.3e-4 \
  --train_dataset $TRAIN_DATASET \
  --val_dataset $VAL_DATASET \
  --slices_column_index $SLICES_COLUMN_INDEX \
  --prop_column_index_list $PROP_COLUMN_INDEX_LIST
