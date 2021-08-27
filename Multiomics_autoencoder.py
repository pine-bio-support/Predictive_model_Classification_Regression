import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Input,Dropout
from keras.models import Model
from matplotlib import pyplot as plt

############################### READING DATASET ##############################

# INPUT NORMALIZED DATASET WITH ONLY 'ID' COLUMN and features COLUMN
data = pd.read_csv("./miRNA+Meth+RNA.csv")  

train,test=train_test_split(data,train_size=0.8,random_state=42)
Xtrain,Xvalidation=train_test_split(train,train_size=0.8,random_state=42)

train_id = train['ID']
train_id=train_id.reset_index(drop=True)


test_id = test['ID']
test_id=test_id.reset_index(drop=True)

train.drop(['ID'], axis=1,inplace=True)
Xtrain.drop(['ID'], axis=1,inplace=True)
Xvalidation.drop(['ID'], axis=1,inplace=True)
test.drop(['ID'], axis=1,inplace=True)


########################### Autoencode: Dimension Reduction ###################

# USING FUNCTIONAL API MODEL
ncol = train.shape[1]
input_dim = Input(shape = (ncol, ))

encoding_dim = 50

# Encoder Layers
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
autoencoder.fit(train, train, epochs = 20,verbose=1, batch_size = 32, shuffle = False, validation_data = (Xvalidation, Xvalidation))
encoder = Model(inputs = input_dim, outputs = encoded) #ASSIGN BOTTLENECK LAYER DATA AS OUTPUT
encoded_input = Input(shape = (encoding_dim, ))

#PLOT
loss = autoencoder.history.history['loss']
val_loss = autoencoder.history.history['val_loss']
epochs = range(20)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("miRNA+Meth+RNA_test.jpeg")

# Prediction
encoded_train = pd.DataFrame(encoder.predict(train))
encoded_train = encoded_train.add_prefix('feature_')
encoded_test = pd.DataFrame(encoder.predict(test))
encoded_test = encoded_test.add_prefix('feature_')

print(encoded_train.shape)
print(encoded_test.shape)

train_out=pd.concat([train_id,encoded_train], axis=1)
test_out=pd.concat([test_id,encoded_test], axis=1)
output=pd.concat([train_out,test_out])
output.to_csv('miRNA+Meth+RNA_relu.csv', index=False) #OUTPUT FEATURE EXTRACTED DATASET
