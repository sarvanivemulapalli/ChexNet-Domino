### AI-Assisted Radiology on Apache Spark - Inference Demo

The purpose of this use case is to demonstrate distributed training of a neural network on a CPU cluster. The goal is to build a model to predict diseases in given chest xray. This is an example for one of the complex image classification problem. In this Demo, we run chestxray inference on few images from x-ray dataset which has 112,120 images with multi class and multi label diseases. This dataset have 50% of "No Findings" images. This Demo helps us to run a quick inference on chestxray images in the form a simple python web application and display the predictions for all 14 classes.

Here is some description to understand about the content inside different chestxray flask webApp version folders,

- **chestxray_webapp.py/logic.py** file has inference code developed with python flask framework and its libraries. It has the paths for labeldata, model, model weights and images.
- **Model** folder has the re-trained model weight files of chestxray which will be used to load and run during the inference.
- **sample_images** is a folder which has the selected set of images from x-ray dataset that we use for our inference test.
- **Templates** folder has html files which has the code for UI design and functionalities for respective functions used in the python file.
- **console.sh/app.sh** file has the command to run the inference code and directs you to the valid web URL.

### Steps to Create Flask Project on Domino
1. Follow the same procedure posted on other [**README**](https://bitbucket.gtie.dell.com/projects/REB/repos/domino/browse/usecases/chestxray/spark/README.md) file but at step-3 when adding/creating files/folders into the new project follow same file structure as below as domino requires the same file structure format to publish the flask webApp without any errors.
```
/mnt
|-- ...
|-- app.sh
|-- chestxray_webapp.py/logic.py
|-- /static
|-- /sample_images
|-- /model files
|-- /templates
```
2. Once you start running the app.sh file for the first time it asks you to hit on `publish` as this helps the domino to make your public with a shared URL like `https://domino.vcse.lab/`. Domino server will be listening on default `port 8888` and `host 0.0.0.0`.
3. Click on publish on the left side pane for using the app location URL to access your flask webApp everytime.

**Note:** Try to use latest flask webApp version files into your projects for reference as they works pretty fine on domino
