import csv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,Activation,Dropout,MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import History 
import matplotlib.pyplot as plt
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
def main():
    xTrain ,yTrain = readCsv('D:\edu\CNN\drowsiness detector project\dataset-dataset.csv')
    print (xTrain.shape[0])
    xTrain = xTrain.astype('float32')
    xTrain /= 255
    model = makeModel()
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        )
    datagen.fit(xTrain)
    H = model.fit_generator(datagen.flow(xTrain,yTrain,batch_size=32),steps_per_epoch=len(xTrain) / 32, epochs=50)
    model.save('DrowsinessModel.hdf5', include_optimizer=False)
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
    
if __name__ == '__main__':
	main()
    
import os
print(os.getcwd())

from keras.models import load_model

md = load_model('DrowsinessModel.hdf5')


D = [178, 179, 181, 180, 175, 171, 170, 168, 162, 154, 153, 154, 153, 152, 144, 137, 138, 135, 133, 136, 139, 143, 147, 151, 152, 154, 152, 150, 154, 161, 168, 173, 181, 184, 184, 180, 179, 178, 175, 173, 172, 169, 166, 155, 150, 148, 147, 148, 145, 141, 134, 128, 124, 124, 126, 129, 136, 142, 150, 155, 159, 160, 164, 168, 171, 174, 183, 186, 187, 186, 186, 184, 178, 173, 171, 169, 158, 153, 154, 149, 141, 137, 133, 130, 131, 126, 124, 124, 123, 121, 125, 131, 141, 150, 158, 164, 169, 174, 177, 180, 189, 192, 187, 180, 175, 170, 164, 160, 159, 158, 150, 145, 144, 139, 130, 127, 123, 120, 115, 111, 109, 110, 109, 108, 113, 121, 128, 137, 147, 155, 164, 173, 182, 189, 196, 199, 184, 172, 163, 160, 156, 149, 139, 131, 130, 124, 124, 122, 118, 115, 107, 100, 100, 96, 92, 91, 89, 90, 97, 107, 115, 124, 134, 145, 157, 170, 182, 192, 199, 202, 174, 163, 153, 146, 141, 139, 140, 141, 140, 140, 145, 138, 122, 110, 98, 91, 87, 87, 88, 88, 87, 86, 89, 94, 104, 112, 124, 137, 151, 164, 176, 186, 197, 200, 155, 142, 133, 134, 139, 149, 159, 165, 155, 147, 136, 114, 88, 72, 62, 56, 57, 61, 67, 73, 77, 80, 85, 89, 96, 103, 114, 128, 144, 158, 171, 180, 194, 198, 132, 126, 135, 164, 189, 189, 165, 141, 90, 68, 49, 37, 37, 44, 41, 34, 35, 36, 37, 41, 49, 62, 76, 85, 92, 98, 107, 120, 137, 154, 169, 180, 193, 197, 123, 121, 157, 189, 194, 163, 131, 96, 58, 52, 37, 41, 45, 56, 51, 41, 35, 35, 37, 36, 39, 44, 63, 80, 92, 95, 113, 119, 143, 156, 171, 182, 192, 201, 142, 151, 158, 152, 132, 91, 69, 72, 77, 65, 57, 66, 66, 75, 76, 61, 33, 49, 57, 41, 35, 38, 55, 81, 88, 96, 109, 116, 146, 161, 180, 189, 193, 202, 165, 172, 137, 83, 57, 62, 90, 127, 129, 103, 84, 71, 75, 86, 92, 51, 24, 50, 66, 50, 46, 47, 61, 97, 97, 95, 94, 105, 148, 174, 190, 186, 194, 203, 145, 136, 99, 59, 59, 108, 155, 174, 168, 117, 98, 127, 104, 113, 95, 43, 18, 37, 52, 52, 54, 49, 64, 101, 114, 96, 79, 86, 127, 178, 197, 188, 194, 203, 108, 77, 61, 72, 108, 162, 202, 206, 202, 157, 162, 184, 116, 101, 83, 31, 31, 42, 52, 59, 57, 50, 77, 111, 122, 107, 93, 83, 101, 171, 196, 191, 195, 204, 68, 54, 67, 109, 155, 184, 202, 208, 215, 199, 193, 201, 113, 92, 61, 48, 43, 53, 59, 60, 52, 51, 89, 117, 117, 110, 102, 86, 94, 170, 196, 191, 197, 205, 53, 72, 92, 127, 167, 193, 201, 211, 216, 211, 186, 137, 89, 79, 60, 58, 55, 61, 62, 58, 59, 68, 97, 114, 108, 103, 101, 96, 114, 177, 200, 198, 199, 206, 95, 119, 120, 132, 162, 200, 208, 217, 223, 217, 200, 134, 82, 60, 52, 62, 57, 57, 54, 55, 73, 92, 106, 111, 107, 110, 121, 132, 154, 184, 196, 197, 201, 209, 125, 138, 124, 128, 145, 186, 201, 220, 218, 220, 219, 185, 112, 73, 45, 57, 56, 50, 52, 70, 94, 109, 114, 113, 115, 126, 148, 163, 174, 188, 196, 198, 204, 208, 136, 136, 122, 118, 128, 147, 163, 182, 201, 205, 213, 199, 165, 138, 104, 94, 87, 91, 101, 113, 121, 124, 125, 127, 127, 137, 157, 170, 179, 188, 195, 201, 207, 211, 159, 155, 153, 143, 144, 134, 138, 142, 140, 139, 147, 148, 145, 131, 123, 126, 124, 130, 136, 136, 128, 122, 125, 131, 134, 145, 161, 173, 183, 190, 195, 205, 209, 212, 173, 169, 173, 157, 157, 145, 149, 140, 131, 131, 133, 131, 132, 120, 126, 138, 137, 139, 140, 136, 129, 125, 130, 137, 134, 146, 158, 167, 182, 191, 194, 204, 208, 212, 189, 185, 185, 169, 171, 172, 179, 167, 155, 172, 166, 148, 148, 151, 147, 135, 133, 134, 135, 136, 134, 134, 136, 139, 144, 154, 159, 164, 182, 195, 198, 205, 209, 213, 197, 192, 193, 188, 188, 190, 191, 179, 183, 185, 175, 166, 163, 157, 152, 151, 145, 144, 142, 141, 141, 143, 145, 148, 155, 162, 164, 166, 184, 199, 202, 207, 211, 214, 200, 195, 197, 197, 198, 198, 198, 190, 189, 196, 190, 172, 164, 162, 156, 151, 151, 149, 146, 144, 144, 148, 154, 158, 160, 167, 172, 175, 190, 202, 204, 208, 212, 215, 203, 199, 202, 202, 202, 201, 202, 196, 190, 188, 187, 178, 177, 170, 164, 162, 155, 156, 156, 156, 156, 158, 161, 163, 169, 177, 185, 191, 202, 209, 209, 213, 212, 216, 201, 200, 199, 201, 203, 202, 200, 197, 192, 189, 185, 181, 177, 174, 170, 168, 166, 164, 163, 162, 165, 168, 170, 175, 179, 186, 193, 199, 203, 208, 211, 212, 215, 218, 201, 200, 200, 201, 203, 203, 200, 197, 196, 194, 190, 186, 183, 180, 177, 175, 174, 172, 171, 170, 173, 176, 177, 183, 190, 196, 202, 205, 208, 211, 213, 212, 215, 218]
X = np.array(D)
X = X.reshape(26,34)
im = np.expand_dims(X, axis = 2)
img = []
img.append(im)# = im


img = np.expand_dims(img, axis = 3)


img_t = np.empty((1,height,width, dims),dtype=np.uint8)


img_t[0] = im


md.predict(img_t)


from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])

layer_output = get_3rd_layer_output([img_t])[0]


X, y = readCsv('D:\edu\CNN\drowsiness detector project\dataset-dataset.csv')
y[0]


final = []
for i in range(0, 2500):
    img_t[0] = X[i]
    layer_op = get_3rd_layer_output([img_t])[0]
    final.append(layer_op.ravel())
    

final = np.array(final)
X_train_svm = final


y_train_svm = []
for i in range(0,2500):
    y_train_svm.append(y[i])

y_train_svm = np.array(y_train_svm)


X_svm = X_train_svm
y_svm = y_train_svm

from sklearn import svm,metrics
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
X_svm = scaler.fit_transform(X_svm)
y_svm = scaler.fit_transform(y_svm)


x_train = X_svm[:1000, :]
y_train = y_svm[:1000, :]
x_test = X_svm[1000:1500, :]
y_test = y_svm[1000:1500, :]

classifier = svm.SVC(gamma=0.001)
classifier.fit(x_train,y_train)

result = classifier.predict(x_test)

labels = [0,1]
conf = confusion_matrix(y_test,result, labels)

score=accuracy_score(y_test,result,normalize=False)

import seaborn as sns
import matplotlib.pyplot as plt
plt.close()
plt.title('Confusion matrix of CNN-SVM')
sns.heatmap(conf,annot = True)
plt.xlabel('True')
plt.ylabel('False')
plt.savefig('Confusion matrix of CNN-SVM')

from sklearn.metrics import classification_report


print(classification_report(y_test,result))
