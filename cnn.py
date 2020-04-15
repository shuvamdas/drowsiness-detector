import csv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,Activation,Dropout,MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import History 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import svm,metrics
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import MinMaxScaler

H = History()
height = 26
width = 34
dims = 1
def readCsv(path):
	with open(path,'r') as f:
		reader = csv.DictReader(f)
		rows = list(reader)
	imgs = np.empty((len(list(rows)),height,width, dims),dtype=np.uint8)
	tgs = np.empty((len(list(rows)),1))
	for row,i in zip(rows,range(len(rows))):
		img = row['image']
		img = img.strip('[').strip(']').split(', ')
		im = np.array(img,dtype=np.uint8)
		im = im.reshape((26,34))
		im = np.expand_dims(im, axis=2)
		imgs[i] = im
		tag = row['state']
		if tag == 'open':
			tgs[i] = 1
		else:
			tgs[i] = 0
	index = np.random.permutation(imgs.shape[0])
	imgs = imgs[index]
	tgs = tgs[index]
	return imgs,tgs	

def makeModel():
	model = Sequential()
	model.add(Conv2D(32, (3,3), padding = 'same', input_shape=(height,width,dims)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, (2,2), padding= 'same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (2,2), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))	
	model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy',metrics=['accuracy'])
	return model



xTrain ,yTrain = readCsv('D:\edu\CNN\drowsiness detector project\dataset-dataset.csv')
print (xTrain.shape[0])
xTrain = xTrain.astype('float32')
xTrain /= 255
print(xTrain.shape)
xTrain_cnn = xTrain[:1000, :]
yTrain_cnn = yTrain[:1000, :]
xTest_cnn = xTrain[1000:, :]
yTest_cnn =  yTrain[1000:, :]
    
    model = makeModel()
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        )
    datagen.fit(xTrain_cnn)
    H = model.fit_generator(datagen.flow(xTrain_cnn,yTrain_cnn,batch_size=32),steps_per_epoch=len(xTrain_cnn) / 32, epochs=50)
    model.save('DrowsinessModel_900_dataset.hdf5', include_optimizer=False)
    plot(H)

 
def plot(H):
    plt.close()
    plt.style.use("ggplot")
    plt.figure()
    N = 50
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    #plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
#plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.show()


from keras.models import load_model

md = load_model('DrowsinessModel_900_dataset.hdf5')
y_pred= md.predict(xTest_cnn)

for i in range(0, 100):
    if y_pred[i][0] > 0.5:
        y_pred[i][0] = 1
    else:
        y_pred[i][0] = 0


labels = [0,1]
conf = confusion_matrix(yTest_cnn,y_pred.round(), labels)

score = accuracy_score(yTest_cnn,y_pred.round(), normalize=False)

import seaborn as sns
import matplotlib.pyplot as plt
plt.close()
plt.title('Confusion matrix of CNN')
sns.heatmap(conf, annot = True)
plt.xlabel('True')
plt.ylabel('False')
plt.savefig('Confusion matrix of CNN')

from sklearn.metrics import classification_report
# plot the first four digits
print(classification_report(yTest_cnn, y_pred.round()))
