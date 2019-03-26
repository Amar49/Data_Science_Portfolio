# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd
# fix random seed for reproducibility
numpy.random.seed(7)

dataset = pd.read_csv('dataset.csv',header=None)
X = dataset.iloc[0:-1,:]
Y= dataset.iloc[-1,:]
X = X.T.values
Y = Y.T.values
print(X.shape)
print(Y.shape)
#print(X,Y)


model = Sequential()
model.add(Dense(12, input_dim=127008, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=5, batch_size=30)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
