python run.py \
    --raw_data_path "raw_mp20_dataset.csv" \
    --cif_column_index 7 \
    --prop_column_index_list 2 3  \
    --structure_json_for_novelty_check "cifs_filtered.json" \
    --complete_train_set_name "mp20_eform_bandgap_nonmetal.csv" \
    --train_output_name "train_data.csv" \
    --val_output_name "val_data.csv"
