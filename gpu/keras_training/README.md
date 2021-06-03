### CheXnet Training using Keras on Single GPU

We are using cheXnet keras code to run the model training on single GPU and calculate the average AUC of all 14 classes

- `auc_callback.py:` This has the AUCROC callback code using keras libraries which will be referred as a function in calculating AUCROC score for all 14 classes.
- `auc_cal_without_hvd.py:` This has the code for calculating the average AUC of all 14 classes from the given trained model weight files.
- `run_keras_auc_cal.sh:`  This is a shell script to run auc_cal_without_hvd.py file using the saved model weights.
- `ChexNet.py:` This has the keras implementation of cheXnet model training and all the model params will be called explicitly inside the code and save the model weights for every epoch with respective validation loss and learningRate.

