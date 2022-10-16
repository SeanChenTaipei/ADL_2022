# ADL HW1

## Environment
```shell
## If you want to run in a virtual env
conda create --name adl-hw1 python=3.9
conda activate adl-hw1
pip install -r requirements.txt
```

## Download pretrained word embedding and Preprocessing
```shell
# To preprocess intent detection and slot tagging datasets
bash ./preprocess.sh
```
Above code will generate a folder named `cache`.

## Download pretrained model weights
```
bash ./download.sh 
```
Pretrained model weights will be downloaded into a folder called `ckpt`.<br>
Weights `ckpt/intent/intent_best_ckpt.tar` and `ckpt/slot/slot_best_ckpt.tar` correspondes to intent classification model and slot tagging model, respectively.


## Training
- Intent classification
```shell
python train_intent.py --batch_size 32 --hidden_size 256 --num_layers 1 --bidirectional True --lr 0.001 --lr_decay 0.5 --num_epoch 30 --warm_up_step 0
```
- Slot tagging
```shell
python train_slot.py --batch_size 16 --hidden_size 256 --num_layers 2 --bidirectional True --lr 0.001 --lr_decay 0.8 --warm_up_step 0 --num_epoch 101
```

## Generate prediction
```shell
##Intent
bash ./intent_cls.sh /data/intent/test.json pred_intent.csv
##Slot
bash ./slot_tag.sh /data/slot/test.json pred_slot.csv

## Slot eval data generation
bash ./slot_tag.sh /data/slot/eval.json eval.csv
```
