from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import yaml
#%%X.shape[1:]

print('Loading Data')
X = pickle.load(open("pkl/X.pickle", "rb"))
y = pickle.load(open("pkl/y.pickle", "rb"))
X = X/255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

with open("model/categories.yaml", 'r') as stream:
    categories = yaml.safe_load(stream)


X.shape[1:]
# %%
n=50
#%%
print('Initializing Model')
model = Sequential()


model.add(Conv2D(32, (3,3) , input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(len(categories)))
model.add(Activation('softmax'))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
#%%

history = model.fit(X_train, y_train, epochs=n, batch_size=64, verbose=1, validation_split=0.2)
model.save("model/model.h5")
model.save_weights('model/modelW.h5')


#%%
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc, "\n")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(n)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('model/tnv.png')
plt.show()
#%%
