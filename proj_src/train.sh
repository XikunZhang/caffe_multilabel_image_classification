#! /bin/bash

rm snapshot*
rm models/*log
rm models/learning_curve*
python train_and_val.py 2>&1 | tee ./models/model_1.log
