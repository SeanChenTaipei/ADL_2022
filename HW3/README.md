# ADL HW3

## Environment
```shell
## If you want to run in a virtual env
conda create --name adl-hw3 python=3.8
conda activate adl-hw3
# pip install -r requirements.txt
```



## Download pretrained model weights
```
bash ./download.sh 
```
Pretrained model weights will be downloaded and unzip into a folder `sum_best_ckpt`.<br>


## Training
- Summarization
```shell
python train.py -h # check out the arguments

python train.py --train_file ./data/train.jsonl --validation_file ./data/public.jsonl \ # file path
                --num_beams 5 --do_sample False --top_k 0 --top_p 1 --temperature 1 \ # generation configs
                --model_name_or_path google/mt5-small --tokenizer_name google/mt5-small \ # huggingface pretrained models
                --per_device_train_batch_size 2 --per_device_eval_batch_size 32 \ # training congifs
                --learning_rate 1e-4 --num_train_epochs 5 --gradient_accumulation_steps 32 \
                --text_column 'maintext' --summary_column 'title' --num_warmup_steps 0 \
                --output_dir ./sum_ckpt \ # output directory
                --with_tracking --ignore_pad_token_for_loss True \
                #--do_rl \ # do rl or not, trains very slow
```


## Generate prediction
```shell
bash ./run.sh /path/to/input.json /path/to/prediction.json
```
