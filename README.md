# Practical Deep Learning Final Project: Applications of Deep Learning in Healthcare: Breast Cancer Classification using Deep Learning models

Term: Spring 2022

+ Team members
	+ Rishav Agarwal (ra3141)
	+ Rachana Dereddy (rd2998)

+ Project summary: The project employs the use of various Deep Learning (Neural Network) models in order to classify Breast Cancer Image appropriately in its two classes of “Non-IDC” and “IDC”. The work proceeds with the Kaggle Breast Histopathology Images, preprocesses it and employs four different neural net models to find out the one that works best. These models were implemented in two different GPU system scenarios: V100 and P100, so that these models can be compared across systems. Our work starts with implementing the VGG16 model, moves on to AlexNet model, and then we create our own model. Finally, we compare all of these models to the method of Transfer Learning using the ResNet50 model. The model that learns best and provides best solution turns out to be the Transfer Learning model with an accuracy close to 85%

ABOUT THE DATASET
Link: https://www.kaggle.com/paultimothymooney/breast-histopathology-images

* The original dataset consisted of 162 whole mount slide images of Breast Cancer (BCa) specimens scanned at 40x. 
* From that, 277,524 patches of size 50 x 50 were extracted (198,738 IDC negative and 78,786 IDC positive). 
* Each patch’s file name is of the format: uxXyYclassC.png —> example 10253idx5x1351y1101class0.png . 
where u is the patient ID (10253idx5), 
* X is the x-coordinate of where this patch was cropped from, 
* Y is the y-coordinate of where this patch was cropped from,  
* C indicates the class where 0 is non-IDC and 1 is IDC.


GitHub Repository Description:

├── (data/)[/data] -> Contains the 
├── doc/
├── figs/
└── output/


Please see each subfolder for a README file.


RESULTS
* For both GPU P100 and V100, we can see from the table provided, VGG16 performs the worst with an accuracy of 50%, while other models go upto 80%.
* The Transfer Learning model with ResNet50 performs the best with an accuracy of 82.78% in P100 and 82.40% for V100 and is the suggested model for these classification.
