# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 test_slot.py --batch_size 16 --hidden_size 256 --num_layers 2 --bidirectional True --test_file "${1}" --ckpt_path ckpt/slot/slot_best_ckpt.tar --pred_file "${2}"