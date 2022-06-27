# Mammographic Severity Classifier

## The Neural Network
This nueral network determine the severity (benign or malignant) of a mammographic lesion on a pateint. The model will predict a value close to 0 if the lesion is predicted to be benign and a 1 if the lesion is predicted to be malignant. Since the model only predicts binary categorical values, the model uses a binary crossentropy loss function and has 1 output neuron. The model uses a standard Adam optimizer with a learning rate of 0.01. The model contains an architecture consisting of:
- 1 Batch Normalization layer
- 1 Input layer (with 6 input neurons and a ReLU activation function)
- 4 Hidden layers (each with 4 neurons and a ReLU activation function)
- 1 Output layer (with 1 output neuron and a sigmoid activation function)

Feel free to further tune the hyperparameters or build upon the model!

## A Note on Accuracy
It should be noted that the neural network (found in the **classifier.py** file) only reaches a testing accuracy of around 85%. Despite hyperparameter tuning (hence a non-standard learning rate of 0.01 and a non-standard batch size of 32), it could not reach test accuracies above 87%. To determine if this issue was only present in the neural network, I tried training other types of models on the same datasets, but, when compared to the neural network, the only model that had similar testing accuracies of around 85%-86% was an SVM (with a linear kernel). The Decision Tree and Random Forest classifiers had the best training accuracies of around 94%, but did not perform as well as the neural network or the SVM in testing accuracy metrics. The comparison can be found in the **comparison.py** file, which will output the respective accuracies of each model and a graph to compare training and testing accuracies. 

The models that were compared on the same data were:
- Neural Network
- Decision Tree Classifier
- Logistic Regression
- Random Forest Classifier
- Support Vector Machines (SVM)
- K Nearest Neighbor

## The Dataset
The dataset can be found at this link: https://www.kaggle.com/datasets/overratedgman/mammographic-mass-data-set. Credit for the dataset collection goes to **Lourens Walters**, **Ovsen**, **Shaurya Jain**, and others on *Kaggle*. It describes the severity (0 or 1) of a lesion based on 5 factors:
- BI-RADS assessment (values 1-5)
- Patient age
- Shape (1 : round, 2 : oval, 3 : lobular, 4 : irregular)
- Margin (1 : circumscribed, 2 : microlobulated, 3 : obscured, 4 : ill-defined, 5 : spiculated)
- Density (1 : high, 2 : iso, 3 : low, 4 : fat-containing)

Note that the initial dataset is unbalanced (this statistic can be found on the data's webpage); it contains 516 instances of benign lesions (encoded as 0's in the model) and 445 instances of malignant lesions (encoded as 1's in the model). This issue is solved within the classifier file using SMOTE, which oversamples the minority class within the dataset.

## Libraries
This neural network was created with the help of the Tensorflow, Imbalanced-Learn, and Scikit-Learn libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit-Learn's Website: https://scikit-learn.org/stable/
- Scikit-Learn's Installation Instructions: https://scikit-learn.org/stable/install.html
- Imbalanced-Learn's Website: https://imbalanced-learn.org/stable/about.html
- Imbalanced-Learn's Installation Instructions: https://pypi.org/project/imbalanced-learn/

