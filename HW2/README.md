# ADL HW1

## Environment
```shell
## If you want to run in a virtual env
conda create --name adl-hw2 python=3.9
conda activate adl-hw2
pip install -r requirements.txt
```

## Download pretrained word embedding


## Download pretrained model weights
```
bash ./download.sh 
```
Pretrained model weights will be downloaded and unzip into 2 folder.<br>
Checkpoint folders `mc_best_ckpt` and `qa_best_ckpt` correspondes to context selection model and question answering model, respectively.


## Training (有空再放)
- Multiple Choice Model
```shell

```
- Question Answering Model
```shell

```

## Generate prediction
```shell
bash ./run.sh /path/to/context.json /path/to/test.json  /path/to/pred/prediction.csv
```