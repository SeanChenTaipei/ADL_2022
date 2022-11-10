# "${1}": path to the context file.
# "${2}": path to the testing file.
# "${3}": path to the output predictions.

# Data Preprocessing for MC testing data
python preprocess_mc.py --context_file ${1} --test_file ${2} --output_dir ./mc_data

# Select relevant context
python mc_test.py --test_file ./mc_data/test_mc.json --ckpt_dir ./mc_best_ckpt --test_batch_size 32 --out_file ./mc_pred_test.json

# QA prediction
python qa_prediction.py --test_file ./mc_pred_test.json --output_dir ${3} --ckpt_dir ./qa_best_ckpt --test_batch_size 32