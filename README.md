# AutoML_CNN_Thesis
Respository for the code used for my thesis on Automated Machine Learning performance against Convolutional Neural Networks in facial emotional recognition.
Autokeras was used as the AutoML algorithm, and Resnet50, InceptionV3 and VGG16 were the CNN architectures that were compared against it on a dataset (found below).
The code and outputs of each model are listed in this repository.
The trained versions of these models on the dataset can be found through the [following link](https://www.mediafire.com/folder/ereyg4tjfqa0m/Trained_models).

Dataset: Kovenko, Volodymyr; Shevchuk, Vitalii (2021), “OAHEGA : EMOTION RECOGNITION DATASET”, Mendeley Data, V2, doi: 10.17632/5ck5zz6f2c.2
https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset

Abstract: Building neural architectures often require expertise in the area of Machine Learning where possessing extensive knowledge into the processes and intricacies of model building is desirable. Automated machine learning (AutoML) algorithms, however, aim to streamline this process and automatically build model architectures based on the dataset given to it. This thesis investigates the performance of Autokeras, an AutoML algorithm, against the Convolutional Neural Networks (CNNs) Resnet50, InceptionV3 and VGG16 for the task of facial emotion recognition (FER). CNNs are often used in FER tasks due to their specialized architectures for spatial learning. Using transfer learning as the main approach to this thesis, the pre-trained ImageNet architectures of the CNNs are leveraged. All pre-trained layers are frozen with exception to the fully connected layer and classification layer. They are then trained on OAHEGA dataset of facial expression emotions. Autokeras is trained on the same dataset, however it is left to its own form of pre-processing and model building. Results indicate superior performance by Autokeras compared to the three CNN architectures, returning an f1 score of 0.82 whilst the CNNs returned scores of 0.41, 0.53 and 0.62 for Resnet50, InceptionV3 and VGG16 respectively. The results showcase a considerable learning gap between Autokeras and the CNN architecures, highlighting AutoML as a potential contender to the traditional machine learning methods that are widely used today.

# Packages and versions used

tensorflow==2.14.0, numpy==1.26.4, scikit-learn==1.2.2, pandas==2.2.1, keras==2.14.0, autokeras==1.0.20, python==3.11.8
