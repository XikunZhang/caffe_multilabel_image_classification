#! /bin/bash

python make_predictions.py
python ~/project/CS446-project_data/npy_to_csv.py subset ./submissions/submission_1.npy ./submissions/subset_submission.csv
