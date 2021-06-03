# Import the required packages
import warnings
# Ignoring the warnings to improve readability of the notebook
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import random
import time
import numpy as np
from math import ceil
from bigdl.optim.optimizer import SGD, SequentialSchedule, Warmup, Poly,\
    Plateau, EveryEpoch, TrainSummary,\
    ValidationSummary, SeveralIteration, Step, L2Regularizer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.storagelevel import StorageLevel
from zoo.common.nncontext import *
from zoo.feature.image.imagePreprocessing import *
from zoo.feature.common import ChainedPreprocessing
from zoo.pipeline.api.keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D, Dropout
from zoo.pipeline.api.keras.metrics import AUC
from zoo.pipeline.api.keras.optimizers import Adam
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.net import Net
from zoo.pipeline.nnframes import NNEstimator, NNImageReader
from zoo.pipeline.api.keras.objectives import BinaryCrossEntropy
from pyspark.sql.types import StringType, ArrayType
import matplotlib.pyplot as plt


# Function to load a ResNet50 model
def get_resnet_model(model_path, label_length):
    full_model = Net.load_bigdl(model_path)
    model = full_model.new_graph(["pool5"])
    inputNode = Input(name="input", shape=(3, 224, 224))
    resnet = model.to_keras()(inputNode)
    flatten = GlobalAveragePooling2D(dim_ordering='th')(resnet)
    dropout = Dropout(0.2)(flatten)
    logits = Dense(label_length, W_regularizer=L2Regularizer(1e-1), b_regularizer=L2Regularizer(1e-1), activation="sigmoid")(dropout)
    lrModel = Model(inputNode, logits)
    return lrModel

# Function to load an Inception model
def get_inception_model(model_path, label_length):
    full_model = Net.load_bigdl(model_path)
    model = full_model.new_graph(["pool5/drop_7x7_s1"])
    inputNode = Input(name="input", shape=(3, 224, 224))
    inception = model.to_keras()(inputNode)
    flatten = GlobalAveragePooling2D(dim_ordering='th')(inception)
    dropout = Dropout(0.25)(flatten)
    logits = Dense(label_length, W_regularizer=L2Regularizer(1e-1), b_regularizer=L2Regularizer(1e-1),\
                   activation="sigmoid")(dropout)
    lrModel = Model(inputNode, logits)
    return lrModel


# Function to load a VGG model
def get_vgg_model(model_path, label_length):
    full_model = Net.load_bigdl(model_path)
    model = full_model.new_graph(["pool5"])
    inputNode = Input(name="input", shape=(3, 224, 224))
    vgg_16 = model.to_keras()(inputNode)
    flatten = GlobalAveragePooling2D(dim_ordering='th')(vgg_16)
    dropout = Dropout(0.25)(flatten)
    logits = Dense(label_length, W_regularizer=L2Regularizer(1e-1), b_regularizer=L2Regularizer(1e-1), activation="sigmoid")(dropout)
    lrModel = Model(inputNode, logits)
    return lrModel

# Function to load a DenseNet model
def get_densenet_model(model_path, label_length):
    full_model = Net.load_bigdl(model_path)
    model = full_model.new_graph(["pool5"])
    inputNode = Input(name="input", shape=(3, 224, 224))
    densenet = model.to_keras()(inputNode)
    flatten = GlobalAveragePooling2D(dim_ordering='th')(densenet)
    dropout = Dropout(0.25)(flatten)
    logits = Dense(label_length, W_regularizer=L2Regularizer(1e-1), b_regularizer=L2Regularizer(1e-1), activation="sigmoid")(dropout)
    lrModel = Model(inputNode, logits)
    return lrModel
    
    
# Learning Rate Scheduler for SGD and Adam optimizer
def get_sgd_optimMethod(num_epoch, trainingCount, batchSize):
    iterationPerEpoch = int(ceil(float(trainingCount) / batchSize))
    warmupEpoch = 10
    warmup_iteration = warmupEpoch * iterationPerEpoch
    initlr = 1e-6
    maxlr = 0.001
    warmupDelta = (maxlr - initlr) / warmup_iteration
    cooldownIteration = (num_epoch - warmupEpoch) * iterationPerEpoch

    lrSchedule = SequentialSchedule(iterationPerEpoch)
    lrSchedule.add(Warmup(warmupDelta), warmup_iteration)
    lrSchedule.add(Plateau("Loss", factor=0.1, patience=1, mode="min", epsilon=0.01, cooldown=0, min_lr=1e-15 ), 
                   cooldownIteration)
    optim = SGD(learningrate=initlr, momentum=0.9, dampening=0.0, nesterov=True,
                leaningrate_schedule=lrSchedule)
    return optim

def get_adam_optimMethod(num_epoch, trainingCount, batchSize):
    iterationPerEpoch = int(ceil(float(trainingCount) / batchSize))
    warmupEpoch = 5
    warmup_iteration = warmupEpoch * iterationPerEpoch
    initlr = 1e-7
    maxlr = 0.0001
    warmupDelta = (maxlr - initlr) / warmup_iteration
    cooldownIteration = (num_epoch - warmupEpoch) * iterationPerEpoch

    lrSchedule = SequentialSchedule(iterationPerEpoch)
    lrSchedule.add(Warmup(warmupDelta), warmup_iteration)
    lrSchedule.add(Plateau("Loss", factor=0.1, patience=1, mode="min", epsilon=0.01, cooldown=0, min_lr=1e-15 ),
                   cooldownIteration)
    optim = Adam(lr=initlr, schedule=lrSchedule)
    return optim
    


#Calculate the AUC-ROC for a disease
def get_auc_for_kth_class(k, df, label_col="label", prediction_col="prediction"):
    get_Kth = udf(lambda a: a[k], DoubleType())
    extracted_df = df.withColumn("kth_label", get_Kth(col(label_col))) \
        .withColumn("kth_prediction", get_Kth(col(prediction_col))) \
        .select('kth_label', 'kth_prediction')
    roc_score = BinaryClassificationEvaluator(rawPredictionCol='kth_prediction',
                                              labelCol='kth_label', metricName="areaUnderROC").evaluate(extracted_df)
    return roc_score
    

# Plot AUC
def plotingAuc(roc_auc_label):
    lists=[]
    lists = sorted(roc_auc_label.items()) 
    label_texts = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
                   "Pneumothorax", "Consolidation","Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
    x, y =zip(*lists)
    label_map = {k: v for v, k in enumerate(label_texts)}
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    sizes = 500 * rng.rand(100)
    colors = ['#005249','#2300A8', '#00A658', '#00A6B8','#00A6BC', '#00AA58','#1805db', '#154406', '#631950','#000000','#850e04','#84b701','#adf802','#042e60']
    plt.ylabel("AUC")
    plt.xlabel("Classes")
    plt.title("AUC for all 14 classes")
    plt.scatter(x, y,  alpha=0.50, color=colors,s=sizes,   cmap='viridis',  marker = '*')
    plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xticklabels(x, rotation=45 )
    return plt


# Evaluate the model and plot AUC
def evaluate(testDF):
    predictionDF = nnModel.transform(testDF).persist(storageLevel=StorageLevel.DISK_ONLY)
    label_texts= ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
                   "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
    label_map = {k: v for v, k in enumerate(label_texts)}
    chexnet_order = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
     "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
    total_auc = 0.0
    roc_auc_label =dict()
    for i in chexnet_order:
        roc_score = get_auc_for_kth_class(label_map[i], predictionDF)
        total_auc += roc_score
        print('{:>12} {:>25} {:>5} {:<20}'.format('ROC score for ', i, ' is: ', roc_score))
        roc_auc_label[i]=(roc_score)
    print("Average AUC: ", total_auc / float(label_length))
    #plt = plotingAuc(roc_auc_label)
    #plt.show()


# Main Program
random.seed(1234)
batch_size = 1024 
num_epoch = 15
# 
#    model_path - Path for the pre-trained model file, data and the location to save the model after training. 
#                 The model path must match the function you are calling (ResNet-50, VGG or DenseNet)
#    image_path - Path to all images
#    label_path - Path to the label file (Data_Entry_2017.csv) available from NIH
#    save_path = Path to save the model and intermediate results 
model_path ="hdfs:///datasets/xray_files/xray/analytics-zoo_resnet-50_imagenet_0.1.0.model" 
image_path = "hdfs:///datasets/xray_files/xray/all_images"
label_path = "hdfs:///datasets/xray_files"
save_path = "hdfs:///datasets/xray_files/save" 

# Get Spark Context
sparkConf = create_spark_conf().setAppName("Chest X-ray Training")
sc = init_nncontext(sparkConf)
spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()


#Load the Data
label_length = 14
label_texts = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax",
               "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
label_map = {k: v for v, k in enumerate(label_texts)}

def text_to_label(text):
    arr = [0.0] * len(label_texts)
    for l in text.split("|"):
        if l != "No Finding":
            arr[label_map[l]] = 1.0
    return arr

getLabel = udf(lambda x: text_to_label(x), ArrayType(DoubleType()))
getName = udf(lambda row: os.path.basename(row[0]), StringType())
imageDF = NNImageReader.readImages(image_path, sc, resizeH=256, resizeW=256, image_codec=1) \
    .withColumn("Image_Index", getName(col('image')))

labelDF = spark.read.load(label_path + "/Data_Entry_2017.csv", format="csv", sep=",", inferSchema="true", header="true") \
        .select("Image Index", "Finding Labels") \
        .withColumn("label", getLabel(col('Finding Labels'))) \
        .withColumnRenamed('Image Index', 'Image_Index')

totalDF = imageDF.join(labelDF, on="Image_Index", how="inner").withColumnRenamed("Finding Labels", "Finding_Labels")

(trainingDF, validationDF) = totalDF.randomSplit([0.7, 0.3])
print("Number of training images: ", trainingDF.count())
print("Number of validation images: ", validationDF.count())


# Load the pretrained model
xray_model = get_resnet_model(model_path, label_length)
# Define the Optimiser
optim_method = get_adam_optimMethod(num_epoch, trainingDF.count(), batch_size)


# Image Pre-Processing
transformer = ChainedPreprocessing(
            [RowToImageFeature(), ImageCenterCrop(224, 224), ImageRandomPreprocessing(ImageHFlip(), 0.5),
             ImageRandomPreprocessing(ImageBrightness(0.0, 32.0), 0.5),
             ImageChannelNormalize(123.68, 116.779, 103.939), ImageMatToTensor(), ImageFeatureToTensor()])

#Save Training and Validation Summary
train_summary = TrainSummary(log_dir="xray_save_summary", app_name="Chest X-ray Training")
val_summary = ValidationSummary(log_dir="xray_save_summary", app_name="Chest X-ray Training")
train_summary.set_summary_trigger("LearningRate", SeveralIteration(50))
train_summary.set_summary_trigger("Loss", SeveralIteration(50))


#Define Classifier
classifier = NNEstimator(xray_model, BinaryCrossEntropy(), transformer) \
            .setBatchSize(batch_size) \
            .setMaxEpoch(num_epoch) \
            .setFeaturesCol("image") \
            .setCachingSample(False) \
            .setValidation(EveryEpoch(), validationDF, [AUC()], batch_size) \
            .setTrainSummary(train_summary) \
            .setValidationSummary(val_summary) \
            .setOptimMethod(optim_method)


# Train Model
nnModel = classifier.fit(trainingDF)


# Evaluate the model and plot AUC accuracy for training Data
print("Evaluating the model on training data:")
evaluate(trainingDF)

SQLContext(sc).clearCache()


# Evaluate the model and plot AUC accuracy for Validation Data
print("Evaluating the model on validation data:")
evaluate(validationDF)


# Save the model for Inference

model_path = save_path + '/xray_model_' + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
nnModel.model.saveModel(model_path + ".bigdl", model_path + ".bin", True)
print('Model saved at: ', model_path)
