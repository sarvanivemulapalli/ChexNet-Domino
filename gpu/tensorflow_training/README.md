### CheXnet Model Training using Tensorflow on single GPU

1. Here we are using cheXnet tensorflow code to run the model training on single GPU and calculate the average AUC of all 14 classes

- `auc_cal_without_hvd.py:` This has the code for calculating the average AUC of all 14 classes from the given weight files.
- `densenet_without_hvd.py:` This has the actual model training code using tensorflow and other library dependencies also we are calling the model params explicitly in this code.
- `run_hvd_auc_cal.sh:` This is a shell script to run auc_cal_without_hvd.py file using the saved model weights.
- `training_labels_new.pkl:` This is a pickle file which divides the training labels from the chestxray dataset. 
- `validation_labels_new.pkl:` This is also a pickle file which divides the validation labels from the chestxray dataset.
- `write_totfrec.py:` This has the code which takes the xray images and training labels pickle file as its input and write the tfrecords data to the output directory which is basically converting the training images from the chestxray dataset to tfrecords.
- `write_totfrec_val.py:` This has the code which takes the xray images and validation labels pickle file as an input and convert that into tfrecords data to the same output directory where the training tfrecords data has been saved and basically this is converting the validation images from the chestxray dataset to tfrecords.

2. The purpose of converting the original xray images to tfrecords is because model training code uses tensorflow libraries and it is easy to read the data as tfrecords.
3. All the experiments/results of cheXnet model training using tensorflow has been recorded at this location,`/home/leela/workspace/cheXnet_tf_hvd` on tesla2_V100_GPU.
