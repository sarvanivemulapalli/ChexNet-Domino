from bigdl.nn.layer import Model
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType, ArrayType

from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.models.image.imageclassification import *
from zoo.pipeline.nnframes import *
from zoo.pipeline.api.net import Net
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.metrics import AUC
from zoo.pipeline.nnframes import NNEstimator
from zoo.pipeline.api.keras.objectives import BinaryCrossEntropy

import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sparkConf = create_spark_conf().setAppName("ChestXray_Inference")
sc = init_nncontext(sparkConf)
spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()
sqlContext = SQLContext(sc)

test_image_path = "file:///mnt/images/00000001_001.png"
label_path = "file:///mnt/trained_model/Data_Entry_2017.csv"

img=mpimg.imread(test_image_path)
imgplot = plt.imshow(img)
plt.show()

label_texts = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
               "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
label_map = {k: v for v, k in enumerate(label_texts)}

def text_to_label(text):
    arr = [0.0] * len(label_texts)
    for l in text.split("|"):
        if l != "No Finding":
            arr[label_map[l]] = 1.0
    return arr

label_length = len(label_texts)

getLabel = udf(lambda x: text_to_label(x), ArrayType(DoubleType()))
getName = udf(lambda row: os.path.basename(row[0]), StringType())
test_imageDF = NNImageReader.readImages(test_image_path, sc, resizeH=256, resizeW=256, image_codec=1)\
                .withColumn("Image Index", getName(col('image')))
imageDF = test_imageDF.withColumnRenamed('Image Index', 'Image_Index')
labelDF = sqlContext.read.option('timestampFormat', 'yyyy/MM/dd HH:mm:ss ZZ')\
            .load(label_path, format="csv", sep=",", inferSchema="true", header="true")\
            .select("Image_Index", "Finding_Labels")\
            .withColumn("label", getLabel(col('Finding_Labels')))\
            .withColumnRenamed('Image Index', 'Image_Index')
labelDF1 = labelDF.withColumnRenamed('Image Index', 'Image_Index')\
            .withColumnRenamed('Finding Labels', 'Finding_Labels')
trainDF = imageDF.join(labelDF1, on="Image_Index", how="inner")


resnet_zoo_model = Net.load("file:///mnt/trained_model/xray_model_2019_04_04_05_31_10.bigdl", "file:///mnt/trained_model/xray_model_2019_04_04_05_31_10.bin")
type(resnet_zoo_model)

def predict(model, inputdf, image_feature_col = "image", batchsize=4):
    """
    Predict output of when inputdf is passed through model
    """
    transformer = ChainedPreprocessing([
        RowToImageFeature(),
        ImageCenterCrop(224, 224),
        ImageChannelNormalize(123.68, 116.779, 103.939),
        ImageMatToTensor(),
        ImageFeatureToTensor()])
    classifier_model = NNModel(model, transformer).setFeaturesCol(image_feature_col)\
                        .setBatchSize(batchsize)
    output = classifier_model.transform(inputdf)
    return output

def show_prediction_output(predDF):
    """
    Display the output size and array
    """
    print(predDF.show(1))
    print("length of prediction array : ", len(predDF.collect()[0].prediction))
    predictions_list = predDF.collect()[0].prediction
    labelList = predDF.collect()[0].label
    print("{:<15} - {:<25} - {:<15}".format('Finding_Labels', 'Prediction', 'Label'))
    print("{:<15} - {:<25} - {:<15}".format('-'*len('Finding_Labels'), '-'*len('Prediction'), '-'*len('Label')))
    for indx in range(0, len(predictions_list)):
        print("{:<15} - {:<25} - {:<15}".format(label_texts[indx], predictions_list[indx], labelList[indx]))
        
        
output = predict(resnet_zoo_model, trainDF)
show_prediction_output(output)

