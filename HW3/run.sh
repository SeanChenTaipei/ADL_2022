# ${1}: path to the input file
# ${2}: path to the output file


# generate predictions

python generation.py --prediction_file ${1} --output_file ${2} \
                     --batch_size 16 --num_beams 5 \
                     --model_name_or_path ./sum_best_ckpt --text_column maintext