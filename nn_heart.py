## adam = the fastest optimizer 

# libraries 
import pandas as pd
from pandas import iterrows
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
import os
%matplotlib inline

#modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report,confusion_matrix

###### CREATING NN WITH DIFFERENT NUMBER OF HIDDEN LAYERS ######

#reading data and presenting a summary of it
df = pd.read_csv('heart.csv')
print(df.shape)
df.describe().transpose()

#creating object of the target variable (target)
target_column = ['target']
predictors = list(set(list(df.columns))-set(target_column)) #setting all other columns than the target column as predictors 
df[predictors] = df[predictors]/df[predictors].max() #normalizing the predictors 
df.describe().transpose() #displaying the normalized data
#all the independent variables have now been scaled/normalized. 

#creating arrays of the independent variables (x) and depdendent variables (y)
X = df[predictors].values
y = df[target_column].values

#splitting the data into train + test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)

#creating neural network consisting of 3 hidden layers with 8 neurons in each layer
#selected the adam as the solver for weight optimization, as it is fast and commonly used 
#selected relu as the activation function 
nn = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
nn1 = MLPClassifier(hidden_layer_sizes=(8,8,8,8), activation='relu', solver='adam', max_iter=500)
nn2 = MLPClassifier(hidden_layer_sizes=(8,8,8,8,8), activation='relu', solver='adam', max_iter=500)
nn3 = MLPClassifier(hidden_layer_sizes=(8,8,8,8,8,8,8,8), activation='relu', solver='adam', max_iter=500)

#now fitting the neural network model to the training data
nn.fit(X_train,y_train)
nn1.fit(X_train,y_train)
nn2.fit(X_train,y_train)
nn3.fit(X_train,y_train)

#now, generating predictions on the training and test dataset 
# model with 3 hidden layers 
predict_train = nn.predict(X_train)
predict_test = nn.predict(X_test)
#model with 4 hidden layers
predict_train1 = nn1.predict(X_train)
predict_test1 = nn1.predict(X_test)
#model with 5 hidden layers
predict_train2 = nn2.predict(X_train)
predict_test2 = nn2.predict(X_test)
#model with 8 hidden layers
predict_train3 = nn3.predict(X_train)
predict_test3 = nn3.predict(X_test)


## EVALUATION PERFORMANCE - NEURAL NETWORK WITH 3 HIDDEN LAYERS
printtrain_image_ame n= [path(confusion_matrix(y_train,predict_train)) #confusion matrix
print(classification_report(y_train,predict_train)) #confusion report results on the training data
#performing with 84% accuracy on TRAINing data. 

# Now, evaluating performance on TEST data 
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
# accuracy of 90%

#########################

## EVALUATION PERFORMANCE - NEURAL NETWORK WITH 4 HIDDEN LAYERS
print(confusion_matrix(y_train,predict_train1)) #confusion matrix
print(classification_report(y_train,predict_train1)) #confusion report results on the training data
#performing with 83% accuracy on TRAINing data. 

# Now, evaluating performance on TEST data 
print(confusion_matrix(y_test,predict_test1))
print(classification_report(y_test,predict_test1))
# accuracy of 89%

########################

## EVALUATION PERFORMANCE - NEURAL NETWORK WITH 5 HIDDEN LAYERS
print(confusion_matrix(y_train,predict_train2)) #confusion matrix
print(classification_report(y_train,predict_train2)) #confusion report results on the training data
#performing with 83% accuracy on TRAINing data. 

# Now, evaluating performance on TEST data 
print(confusion_matrix(y_test,predict_test2))
print(classification_report(y_test,predict_test2))
# accuracy of 92%

#########################

## EVALUATION PERFORMANCE - NEURAL NETWORK WITH 8 HIDDEN LAYERS
print(confusion_matrix(y_train,predict_train3)) #confusion matrix
print(classification_report(y_train,predict_train3)) #confusion report results on the training data
#performing with 85% accuracy on TRAINing data. 

# Now, evaluating performance on TEST data 
print(confusion_matrix(y_test,predict_test3))
print(classification_report(y_test,predict_test3))
# accuracy of 89%


#################### SVM CLASSIFICATION ####################
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train,y_train)

#making predictions 
y_pred = svclassifier.predict(X_test)

#evaluating
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

################## CNN - DIAGNOSIS OF CHEST X-RAY DATASET SAMPLE ##################

#libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#creating an object of sequential class. i want the cnn layers to be sequential
classifier = Sequential()

#convolution with 32 filters with 3x3 shape, 200x200 resolutional and 3 = colour image.
classifier.add(Conv2D(32,(3,3),input_shape = (200,200,3),activation ='relu'))

#performing pooling operation. 2x2 matrix. minimum pixel loss and precise region where features are located 
classifier.add(MaxPooling2D(pool_size=(2,2)))

#converting all pooled images to continuous vector through flattening
#bascially, converting the pooled image pixels into one dimensional single vector
classifier.add(Flatten())

#creating and adding a fully connected layer
classifier.add(Dense(units = 128, activation ='relu'))

#output layer. final layer contains only 1 node and using sigmoid activation function for final layer
classifier.add(Dense(units = 1, activation ='sigmoid'))

#now its built and i need to compile it
classifier.compile(optimizer ='adam',loss='binary_crossentropy',metrics=['accuracy'])
#loss parameter is just choosing the loss function.

classifier.summary()



### NOW FITTING MY MODEL TO MY DATASET ###
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets
from os import listdir
from matplotlib import image
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import asarray
#load csv data 
cnn1 = pd.read_csv('sample_labels.csv')
print(cnn1.shape)
cnn1.describe().transpose()

#load images
path = os.getcwd() + '/images/'

photos, labels = list(), list()
# enumerate files in the directory
for file in listdir(path):
	# determine class

	# load image
	photo = load_img(path + file, target_size=(200, 200))
	# convert to numpy array
	photo = img_to_array(photo)
	# store
	photos.append(photo)
	labels.append(file)
# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)

### REARRANGE - så labels og billederne passer sammen
# loop igennem photo liste, find label fra csv liste
cnn1['data_order'] = np.nan
for idx, filename in enumerate(labels):
	for idx2, name in enumerate(cnn1['Image Index']):
		if filename == name:
			cnn1.at[idx2, 'data_order'] = idx



#sorterer cnn1 så de står i rigtig rækkefølge, ligesom photos listen. 
cnn1 = cnn1.sort_values('data_order')
cnn1 = cnn1.reset_index()

#labels 
labs = cnn1['Finding Labels']

#se photos datasæt 
photos.shape

#photos skal bruges til predicte!! 

#https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/


## DOWNSIZING DATASET - since it showed (very) poor results when using the full dataset

#  test = labs.isin(['No Finding', 'Infiltration']) == True
#  #fjern alle rows med False 
#  #fjern alle samme i billederne 

# #the rows with True 
# true_rows = labs.loc[labs.isin(['No Finding', 'Infiltration']) == True].index.values

# #removing .. 

# test[test.name != 'True']
# test.drop('False')

# only_true_labs = labs[labs['Finding Labels'] == True]

# only_true_labs = labs.loc[labs['Finding Labels'], :]

# labs.drop(labs[labs['Finding Labels'] == False].index, inplace=True)

# test.drop(test['Finding Labels' == False].index, inplace=True)

# teeest = test[test('Finding Labels' == False).all]

#creating empty list
labs_list = labs.tolist()

#find all rows with 'no finding' and 'infiltration' and add to new list 
indices = [idx for idx, i in enumerate(labs_list) if i in ['No Finding', 'Infiltration']] 
new_labs = [i for idx, i in enumerate(labs_list) if idx in indices]

#only keep the same rows as in the indices (i.e. only rows containing 'no finding' and 'infiltration')
photo_test = photos[indices, :, :, :]

#see the length lists of labels and photos - checking they match each other  
len(indices)
photo_test.shape

#SOON, ready to test 

#transforming the labels into numbers instead of strings 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(new_labs) #res? 
new_labs = le.transform(new_labs) 

#splitting dataset into 80/20 train/test 
X = asarray(photo_test)
y = asarray(new_labs)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape);print(X_test.shape)

#training the model
classifier.fit(X_train,y_train,epochs = 10,batch_size=64)

#on full dataset, with 244 predictions and 5606 different chest X-ray images, it showed a horrible accuracy. not even worth mentioning, but here goes: an almost a negative accuracy
#however, on downsampled dataset with only 2 predictors and 3547 different chest X-ray images, it showed an accuracy of 86 %
# when classifying between no finding or infiltration in the chest x rays images. 





#### SVM - CHEST X-RAY IMAGES ### 

# If time, try SVM to classify X-Ray images: 

#need to reshape the data to 2 dimensions instead of 4 
# nsamples, nx, ny = train_dataset.shape
# d2_train_dataset = train_dataset.reshape((nsamples,nx*ny)) 
 
# svm = SVC(kernel='linear', probability=True, random_state=42)
# svm.fit(X_train, y_train)