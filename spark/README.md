### AI-assisted Radiology using Distributed Deep Learning on Apache Spark and Analytics Zoo

This is one of the usecases implemented on domino to demonstrate how we can we can build an end to end deep learning pipeline on Spark leveraging the Analytics Zoo for an image processing problem. Distributed Spark worker nodes are used to train our deep learning model at scale. We used the Chest Xray dataset released by the National Health Institute to develop an AI models to diagnose pneumonia, emphysema, and other thoracic pathologies from chest x-rays. Using the Stanford University CheXNet model as inspiration, we explore ways of developing accurate models for this problem on a distributed Spark cluster. We explore various neural network topologies to gain insight into what types of modelsneural networks scale well in parallel and reduce to improve training time from days to hours.

In this usecase we have done Training, Inference, Demo using chestxray dataset by utilizing Domino and it's infrastructure.

[**Training**](https://bitbucket.gtie.dell.com/projects/REB/repos/domino/browse/usecases/chestxray/training)

- **chestxray_training.py:** This has model training code integrated with AUCROC code which helps in calculating the average AUC for all 14 classes once training is done.
- **run_training.sh:** This is a shell script used to run the model training python file by using spark-submit with required spark params.

[**Inference**](https://bitbucket.gtie.dell.com/projects/REB/repos/domino/browse/usecases/chestxray/inference)

- **chestxray_inference.py:** This is used to run the prediction(inference) diseases from chestxray images.
- **run_inference.sh:** This is a shell script to run the inference python file with some required spark params.

[**Flask WebApp**](https://bitbucket.gtie.dell.com/projects/REB/repos/domino/browse/usecases/chestxray/flask_webApp)

This is the webApp implemented on domino for this chestxray usecase and this was developed using python flask framework which is extensively supported by domino platform. To explore more about things we worked on this webApp refer to this [**README**](https://bitbucket.gtie.dell.com/projects/REB/repos/domino/browse/usecases/chestxray/flask_webApp). We have developed 4 versions of this webApp and every version has some minor changes from the previous ones.


### Steps to Create Project and Run on Domino

1. Login with your user credentials into [**domino.vcse.lab**](https://domino.vcse.lab/login) environment.
2. On the top navigation bar, click on projects and click on new project on the top right. Name your project and click create project.
3. Click on files tab on the left side pane add your required files/code.
4. Under settings on left side pane click on hardware tier drop down to select default(CPU) as your preferred infrastruture.
5. Click on compute environment drop down to select the stable virtual environment which is CDH5_V3.
6. Under Access and Sharing choose the visibility of your project to public.
7. Under Integration tab, select Apache Spark to YARN and add the below details into those tabs and click save.
   - **Kerberos:** No Kerberos Authenctication
   - **Hadoop user name:** hdfs/root
   - **Custom /etc/host entries:** add IP's and hostnames of all the workers in platinum cluster
   - **spark Configuration Option:** 
```
spark.yarn.appMasterEnv.DL_ENGINE_TYPE              mklblas
spark.yarn.appMasterEnv.MKL_DISABLE_FAST_MM         1
spark.yarn.appMasterEnv.KMP_BLOCKTIME               0
spark.yarn.appMasterEnv.OMP_WAIT_POLICY             passive
spark.yarn.appMasterEnv.OMP_NUM_THREADS             1
spark.shuffle.reduceLocality.enabled                false
spark.shuffle.blockTransferService                  nio
spark.scheduler.minRegisteredResourcesRatio         1.0
spark.rpc.message.maxSize                           64
spark.jars                                          /opt/Intel/lib/analytics-zoo-bigdl_0.7.2-spark_2.2.0-0.4.0-jar-with-dependencies.jar
spark.executor.memory                               300g
spark.executor.cores                                32
spark.driver.memory                                 300g
spark.executor.instances                            4
spark.submit.pyFiles                                /opt/Intel/lib/analytics-zoo-bigdl_0.7.2-spark_2.2.0-0.4.0-python-api.zip
spark.dynamicAllocation.enabled                     false
spark.speculation                                   false
spark.serializer                                    org.apache.spark.serializer.JavaSerializer
```
8. Once adding all these settings, run the code using shell script or related file by using run/launch notebook button on top right corner you can see when you select the file to run experiment.
