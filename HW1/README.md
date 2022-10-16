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
bash preprocess.sh
```
Above code will generate a folder named **cache**.

## Download pretrained model weights
```
bash ./download.sh 
```
Pretrained model weights will be downloaded into a folder **ckpt**.

## Intent detection
```shell
python train_intent.py
```
