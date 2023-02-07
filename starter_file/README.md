# Capstone Project - Heart Disease Prediction

This is the final project for the Udacity Machine Learning Nanodegree program.In this project,created two experiments; one using Microsoft Azure Machine Learning Hyperdrive package, and another using Microsoft Azure Automated Machine Learning with the Azure Python SDK.

The best models from the two experiments were compared based on the primary metric (AUC Weighted) and the best performing model was deployed and consumed using a web service.
<img width="696" alt="Screen Shot 2023-01-29 at 6 51 59 PM" src="https://user-images.githubusercontent.com/46094082/215366592-3bb79347-a5a0-44cb-9bff-abe84f8b13d5.png">


## Dataset

Cardiovascular diseases (CVDs) are the number one cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.


This Dataset consists of 12 columns, which will help us predict HeartDisease.
1. Age: age of the patient [years]
2. Sex: sex of the patient [M: Male, F: Female]
3. ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
4. RestingBP: resting blood pressure [mm Hg]
5. Cholesterol: serum cholesterol [mm/dl]
6. FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
7. RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
8. MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
9. ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
10. Oldpeak: oldpeak = ST [Numeric value measured in depression]
11. ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
12. HeartDisease: output class [1: heart disease, 0: Normal]

### Task
This is a binary classification problem, where the outcome 'HeartDisease' will either be '1' or '0'. In this experiment, we will use Hyperdrive and AutoML' to train models on the dataset based on the AUC Weighted` metric. We will then deploy the model with the best performance and interact with the deployed model.

### Access
Dataset is created uploading the downloaded datafile from kaggle and registered the dataset to the datastore.

ds = TabularDatasetFactory.from_delimited_files(['https://raw.githubusercontent.com/sireeshag09/udacity-capstone/main/heart.csv'])

## Automated ML
An AutoML is built on the HeartDisease dataset to automatically train and tune machine learning algorithms at various hyperparameter tuning and feature selection for an optimal selection of a model that best fits the training dataset using a given target metric.

Below is the automl settings and configuration specified for automl to set and run that yielded Best accuracy.
<img width="750" alt="Auto ML Config and settings " src="https://user-images.githubusercontent.com/46094082/216803778-06fcd604-81b0-49cd-9c35-faa199b66837.png">

### Results
**Run Details**


<img width="1292" alt="Run Details widget" src="https://user-images.githubusercontent.com/46094082/216803582-92220f20-9846-4638-b93b-c9f0db296078.png">

**Best Model**

The best performing model is the VotingEnsemble with an AUC_weighted value of 0.9380 and accuracy of 0.88568. A voting ensemble is a technique to improve model performance that balances out the individual weaknesses of the considered classifiers.

<img width="1272" alt="AutoML Best Model" src="https://user-images.githubusercontent.com/46094082/216803838-66e7ac48-a523-42f3-8933-0b96b1c8f7a7.png">

## Hyperparameter Tuning
The key steps for HyperDrive Tuning involves search space,Sampling Method,Primary metric and early termination policy. In this project Random Sampling search space adopted with the intention to randomly sample hyperparameter values from a defined search space without incurring high computational cost.Even though search space supports both discrete and continuous values as hyperparameter values but the search space is set to discrete for Regularization parameter, C, and Max-iter because it achieved the best accuracies compared to the accuracies obtained from the model when the continuous search space was used.

**Run Details**

<img width="1219" alt="Hyperparameter Run details" src="https://user-images.githubusercontent.com/46094082/216807690-cf887e99-68cf-4bb7-8acc-6863413452ac.png">

**Hyperparameters**

Model training can be controlled through hyperparameters and run experiments in parallel to efficiently optimize hyperparameters.

The best performing model using HyperDrive has Parameter Values as batchsize = 20, frequency = 600. 
The AUC_weighted of the Best Run is 0.9415233415233415 with an accuracy of 0.89130434
<img width="1600" alt="Hyperparameter Run Visualization" src="https://user-images.githubusercontent.com/46094082/216807543-5208bee7-9ad9-46f2-a576-550a92e5e250.png">

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

Even though HyperDrive Model performed slightly better than AutoML model for HeartDisease dataset, AutoML was choosen to deploy as a web service. Below are the steps performed for deploying a model in Azure ML Studio

1.Register the best model

2.Setup inference configuration for the web service containing the model to be used for deploying the model.

3.Entry script to submit data to a deployed web service and passes it to the model and returns the model response to the client application.

4.Choose a compute target

5.Deploy the model to the compute target

6.Test the web service by sending requests using input data and getting responses for predictions.

**Active Endpoint**

<img width="973" alt="Active Endpoint" src="https://user-images.githubusercontent.com/46094082/216807842-dd0c82cd-a697-4b56-9846-5472232ec2c1.png">

## Screen Recording

 Link to the screen recording of the capstone project in action demonstarting the working deployed model.
- https://youtu.be/E9kZBY8aXT8

## Standout Suggestions

To detect anomalies and visulaize performance

1.Enable Application Insights

2.Enable logging

AutoML results can be improved adopting any of these 

1.Increased experiment timeout duration

2.Exploring other primary metrics like f1 score,precision and recall.

3.Optimizing other AutoML configurations.

HyperDrive results can be improved as well choosing different algorithm such as Xgboost or by setting different termination policy and sampling methods.
