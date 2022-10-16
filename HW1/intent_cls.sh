# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 test_intent.py --test_file "${1}" --ckpt_path ckpt/intent/intent_best_ckpt.tar --pred_file "${2}" --hidden_size 256 --num_layers 1 --bidirectional True --batch_size 20