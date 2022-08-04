# Mammographic Severity Classifier

## The Neural Network
This neural network determines the severity (benign or malignant) of a mammographic mass lesion on a patient. The model will predict a value close to 0 if the mass lesion is predicted to be benign and a 1 if the mass lesion is predicted to be malignant. Since the model only predicts binary categorical values, the model uses a binary crossentropy loss function and has 1 output neuron. The model uses a standard Adam optimizer with a learning rate of 0.01. The model contains an architecture consisting of:
- 1 Batch Normalization layer
- 1 Input layer (with 6 input neurons and a ReLU activation function)
- 4 Hidden layers (each with 4 neurons and a ReLU activation function)
- 1 Output layer (with 1 output neuron and a sigmoid activation function)

Feel free to further tune the hyperparameters or build upon the model!

## Other Models
It should be noted that the neural network (found in the **classifier.py** file) only reaches a testing accuracy of around 85%. Despite hyperparameter tuning (hence a non-standard learning rate of 0.01 and a non-standard batch size of 32), it could not reach test accuracies above 87% (occasionally the model does much worse or better than average). To determine if this issue was only present in the neural network, I tried training other types of models on the same datasets, but, when compared to the neural network, the only models that had similar testing accuracies of around 85%-86% were an SVM (with a linear kernel) and an XGBoost Classifier (with 5000 estimators and a learning rate of 0.001). Overall, the SVM seemed to consistently have the best test accuracy. The Decision Tree and Random Forest classifiers had the best training accuracies of around 94%, but did not perform as well as the neural network, SVM, or XGBoost Classifier in testing accuracy metrics. The comparison discussed here can be found in the **comparison.py** file, which will output the respective training and testing accuracies of each model and a graph to compare those accuracies across models. 

The models that were compared on the same data were:
- Neural Network
- Decision Tree Classifier
- Logistic Regression
- Random Forest Classifier
- Support Vector Machines (SVM)
- K Nearest Neighbor
- XGBoost Classifier

## The Dataset
The dataset can be found at this link: https://www.kaggle.com/datasets/overratedgman/mammographic-mass-data-set. Credit for the dataset collection goes to **Lourens Walters**, **Ovsen**, **Shaurya Jain**, and others on *Kaggle* and the *Institute of Radiology of the University Erlangen-Nuremberg*. It describes the severity (0 or 1) of a lesion based on 5 factors:
- BI-RADS assessment (values 1-5)
- Patient age
- Mass shape (1 : round, 2 : oval, 3 : lobular, 4 : irregular)
- Mass margin (1 : circumscribed, 2 : microlobulated, 3 : obscured, 4 : ill-defined, 5 : spiculated)
- Mass density (1 : high, 2 : iso, 3 : low, 4 : fat-containing)

Mass shape, mass margin, and mass density are BI-RADS attributes. Note that the initial dataset is unbalanced (this statistic can be found on the data's webpage); it contains 516 instances of benign lesions (encoded as 0's in the model) and 445 instances of malignant lesions (encoded as 1's in the model). This issue is solved within the classifier using SMOTE, which oversamples the minority class within the dataset.

## Potential Applications
The neural network, SVM, and other models in this project could hypothetically advise a physician on whether or not to perform a breast biopsy on a patient. Misinterpretation of mammographs is the primary cause of unnecessary breast biopsies (biopsies that result in benign outcomes). By predicting the severity of a lesion, the models in this project can hypothetically help physicians discern if they should conduct a breast biopsy on a patient; if the models predict values close to 0, they suggest the lesion is likely benign and thus not worthy of a complete biopsy, while if they predict values close to 1, they suggest the lesion is likely malignant and thus worthy of a biopsy.  

## Libraries
The neural network and other models were created with the help of the Tensorflow, Imbalanced-Learn, and Scikit-Learn libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit-Learn's Website: https://scikit-learn.org/stable/
- Scikit-Learn's Installation Instructions: https://scikit-learn.org/stable/install.html
- Imbalanced-Learn's Website: https://imbalanced-learn.org/stable/about.html
- Imbalanced-Learn's Installation Instructions: https://pypi.org/project/imbalanced-learn/
- XGBoost's Website: https://xgboost.readthedocs.io/en/stable/#
- XGBoost's Installation Instructions: https://xgboost.readthedocs.io/en/stable/install.html

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual medical use or application in any way.
