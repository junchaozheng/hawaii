# READ ME
# Description and Instruction for HW1 Code

1.File Description:

1) data augmentation.py
It will dump a new file which contains 90k data after applying data augmentation.

2) mnist_mode.py
It will training model on both labeled data and unlabeled data. It will dump model for every 20 epoches. 

3) mnist_model.sh (optional)
It is a bash command which helps to run scripts on HPC

4) mnist_result.py
It takes the input from our best model file (model_1488643330_320.p) and test.p to make a prediction in a csv file for Kaggle submission.

5) data:
It contains train_label.p, validation.p, test.p, train_unlabel.p and train_labeled_allmethod.p(This is the data file after using data augmentation)

6) model:
It contains the best model while we were training

2. Insturction
First, run the following command to reproduce our work. Make sure that you install torchvision. (pip install torchvision)
Then, run the following command.

```
python mnist_model.py
```

The last step is runing the following command to create a prediction in csv file.

```
python mnist_result.py
```
