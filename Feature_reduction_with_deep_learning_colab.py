import pandas as pd
import sys
import csv
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Input,Dropout
from keras.models import Model
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier as ANN


# INPUT NORMALIZED DATASET WITH ONLY 'ID' COLUMN and features COLUMN

#https://drive.google.com/drive/u/1/my-drive

#Load data from github
url1 = 'https://raw.githubusercontent.com/pine-bio-support/Predictive_model_Classification_Regression/main/train_data.txt'
train = pd.read_table(url1)

url2 = 'https://raw.githubusercontent.com/pine-bio-support/Predictive_model_Classification_Regression/main/test_data.txt'
test = pd.read_table(url2)


#train = pd.read_csv("train_data.txt", sep='\t')

#test = pd.read_csv("test_data.txt", sep='\t')


train_id = train['id']
train_id=train_id.reset_index(drop=True)


test_id = test['id']
test_id=test_id.reset_index(drop=True)



################### split if training data into 5 folds ##########

Xtrain,Xvalidation=train_test_split(train,train_size=0.8,random_state=42)

################### Drop id column from matrices #################

train.drop(['id'], axis=1,inplace=True)
Xtrain.drop(['id'], axis=1,inplace=True)
Xvalidation.drop(['id'], axis=1,inplace=True)
test.drop(['id'], axis=1,inplace=True)


################ Autoencode: Dimension Reduction ###################

# USING FUNCTIONAL API MODEL

ncol = train.shape[1]

input_dim = Input(shape = (ncol, ))

encoding_dim = 50

encoded = Dense(500, activation = 'relu')(input_dim)
encoded = Dropout(0.5)(encoded) #DROUPOUT
encoded = Dense(100, activation = 'relu')(encoded)
encoded = Dense(encoding_dim, activation = 'relu')(encoded) # BOTTLENECK LAYER

# Decoder Layers
decoded = Dense(100, activation = 'relu')(encoded)
decoded = Dense(500, activation = 'relu')(decoded)
decoded = Dense(ncol, activation = 'sigmoid')(decoded)

# Combine Encoder and Deocder layers
autoencoder = Model(inputs = input_dim, outputs = decoded)

# Compile the Model
autoencoder.compile(optimizer = 'Adam', loss = 'binary_crossentropy')
#autoencoder.fit(train, train, epochs = 20,verbose=1, batch_size = 32, shuffle = False, validation_data = (Xvalidation, Xvalidation))
autoencoder.fit(train, train, epochs = 20,verbose=1, batch_size = 32, shuffle = False, validation_data = (Xvalidation, Xvalidation))
encoder = Model(inputs = input_dim, outputs = encoded) #ASSIGN BOTTLENECK LAYER DATA AS OUTPUT
encoded_input = Input(shape = (encoding_dim, ))

# Loss and PLOT
loss = autoencoder.history.history['loss']
val_loss = autoencoder.history.history['val_loss']
epochs = range(20)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("train_test_loss.jpeg")

# Prediction
encoded_train = pd.DataFrame(encoder.predict(train))
encoded_train = encoded_train.add_prefix('feature_')
encoded_test = pd.DataFrame(encoder.predict(test))
encoded_test = encoded_test.add_prefix('feature_')


print (encoded_train.shape)
print (encoded_test.shape)

train_out=pd.concat([train_id,encoded_train], axis=1)
test_out=pd.concat([test_id,encoded_test], axis=1)

####### OUTPUT FEATURE EXTRACTED DATASET #########

# Import Drive API and authenticate.
#from google.colab import drive

# Mount your Drive to the Colab VM.
#drive.mount('/gdrive')

#train_out.to_csv('/gdrive/deep_train_mat.csv', index=False)
#test_out.to_csv('/gdrive/deep_test_mat.csv', index=False)


from google.colab import files

train_out.to_csv('deep_train_mat.csv', index=False)
test_out.to_csv('deep_test_mat.csv', index=False)
files.download("deep_train_mat.csv")
files.download("deep_test_mat.csv")
