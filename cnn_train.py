# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, re, time, math, tqdm, itertools
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import keras
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dense
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint
# check the available data


network_data = pd.read_csv('new1.csv')

# check the shape of data
print(network_data.shape)

# check the number of rows and columns
print('Number of Rows (Samples): %s' % str((network_data.shape[0])))
print('Number of Columns (Features): %s' % str((network_data.shape[1])))

network_data.head(4)

# check the columns in data
network_data.columns

# check the number of columns
print('Total columns in our data: %s' % str(len(network_data.columns)))

network_data.info()

# check the number of values for labels
network_data['Class'].value_counts()


#Data Visualizations
#After getting some useful information about our data, we now make visuals of our data to see how the trend in our data goes like. The visuals include bar plots, distribution plots, scatter plots, etc.

# make a plot number of labels
sns.set(rc={'figure.figsize':(12, 6)})
plt.xlabel('Attack Type')
sns.set_theme()
ax = sns.countplot(x='Class', data=network_data)
ax.set(xlabel='Attack Type', ylabel='Number of Attacks')
plt.show()


#Data Preprocessing
#Data preprocessing plays an important part in the process of data science, since data may not be fully clean and can contain missing or null values. In this step, we are undergoing some preprocessing steps that will help us if there is any null or missing value in our data.

# check for some null or missing values in our dataset
print(network_data.isna().sum().to_numpy())
# drop null or missing columns
cleaned_data = network_data.dropna()
cleaned_data.isna().sum().to_numpy()

###Label Encoding

# encode the column labels
label_encoder = LabelEncoder()
cleaned_data['Class']= label_encoder.fit_transform(cleaned_data['Class'])
cleaned_data['Class'].unique()

# check for encoded labels
print(cleaned_data['Class'].value_counts())

'''
Shaping the data for CNN
For applying a convolutional neural network on our data, we will have to follow following steps:

Seperate the data of each of the labels
Create a numerical matrix representation of labels
Apply resampling on data so that can make the distribution equal for all labels
Create X (predictor) and Y (target) variables
Split the data into train and test sets
Make data multi-dimensional for CNN
Apply CNN on data'''
# make 3 seperate datasets for 3 feature labels
data_1 = cleaned_data[cleaned_data['Class'] == 0]
data_2 = cleaned_data[cleaned_data['Class'] == 1]

# make benign feature
y_1 = np.zeros(data_1.shape[0])
y_benign = pd.DataFrame(y_1)

# make DDOS feature
y_2 = np.ones(data_2.shape[0])
y_bf = pd.DataFrame(y_2)



# merging the original dataframe
X = pd.concat([data_1, data_2], sort=True)
y = pd.concat([y_benign, y_bf], sort=True)
print(y_1, y_2)
print(X.shape)
print(y.shape)

# checking if there are some null values in data
X.isnull().sum().to_numpy()


#Data Argumentation
#Ti avoid biasing in data, we need to use data argumentation on it so that we can remove bias from data and make equal distributions.

from sklearn.utils import resample

data_1_resample = resample(data_1, n_samples=4873, 
                           random_state=123, replace=True)
data_2_resample = resample(data_2, n_samples=3858, 
                           random_state=123, replace=True)
train_dataset = pd.concat([data_1_resample, data_2_resample])
train_dataset.head(2)

# viewing the distribution of intrusion attacks in our dataset 
plt.figure(figsize=(10, 8))
circle = plt.Circle((0, 0), 0.7, color='white')
plt.title('Intrusion Attack Type Distribution')
plt.pie(train_dataset['Class'].value_counts(), labels=['0', '1'], colors=['blue', 'magenta', 'cyan'])
p = plt.gcf()
p.gca().add_artist(circle)



#Making X & Y Variables (CNN)
test_dataset = train_dataset.sample(frac=0.1)
target_train = train_dataset['Class']
target_test = test_dataset['Class']
target_train.unique(), target_test.unique()


y_train = to_categorical(target_train, num_classes=2)
y_test = to_categorical(target_test, num_classes=2)

#Data Splicing
#This stage involves the data split into train & test sets. The training data will be used for training our model, and the testing data will be used to check the performance of model on unseen dataset. We're using a split of 80-20, i.e., 80% data to be used for training & 20% to be used for testing purpose.

train_dataset = train_dataset.drop(columns = ["Class"], axis=1)
test_dataset = test_dataset.drop(columns = ["Class"], axis=1)
# making train & test splits
X_train = train_dataset.iloc[::-1].values
X_test = test_dataset.iloc[::-1].values


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# reshape the data for CNN
X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
print(X_train.shape, X_test.shape)

# making the deep learning function
def model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(35, 1)))
    model.add(BatchNormalization())
    
    # adding a pooling layer
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(35, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(35, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = model()
model.summary()
logger = CSVLogger('logs.csv', append=True)
his = model.fit(X_train, y_train, epochs=500, batch_size=32, 
          validation_data=(X_test, y_test), callbacks=[logger])



# check the model performance on test data
scores = model.evaluate(X_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
model.save('model1.h5')
# check history of model
history = his.history
history.keys()

epochs = range(1, len(history['loss']) + 1)
acc = history['accuracy']
loss = history['loss']
val_acc = history['val_accuracy']
val_loss = history['val_loss']

# visualize training and val accuracy
plt.figure(figsize=(10, 5))
plt.title('Training and Validation Accuracy (CNN)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(epochs, acc, label='accuracy')
plt.plot(epochs, val_acc, label='val_acc')
plt.legend()

# visualize train and val loss
plt.figure(figsize=(10, 5))
plt.title('Training and Validation Loss(CNN)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, loss, label='loss', color='g')
plt.plot(epochs, val_loss, label='val_loss', color='r')
plt.legend()




        
        