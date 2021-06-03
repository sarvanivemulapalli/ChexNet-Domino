### CheXnet Training on GPU
Here we approached with two ways of training cheXnet code on single GPU 

- [**Tensorflow + GPU**](https://bitbucket.gtie.dell.com/projects/REB/repos/domino/browse/usecases/chestxray/gpu/tensorflow_training)
- [**Keras + GPU**](https://bitbucket.gtie.dell.com/projects/REB/repos/domino/browse/usecases/chestxray/gpu/keras_training)

We are using Tesla V100 as our single GPU as a base infrastructure and tensorflow, keras libraries as part of our code.

Using tensorflow code on single GPU helps us to complete the model training in less than 2.5hrs and the average AUC we observed for all 14 classes is around 69% but using keras code on single GPU completes the model training in ~5hrs and the average AUC for all 14 classes is around 80%
