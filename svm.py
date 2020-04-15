

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm, metrics
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score


from subprocess import check_output
#print(check_output(["ls", "../SVM"]).decode("utf8"))


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
x_Train,y_Train =readCsv('D:\edu\CNN\drowsiness detector project\dataset-dataset.csv')


xTrain = x_Train[:1000, :]
yTrain = y_Train[:1000, :]

xTest = x_Train[1000:1500, :]
yTest = y_Train[1000:1500, :]

print(xTrain.shape)
print(xTrain[0,0])

nsamples, nx, ny,k = xTrain.shape
print( xTrain.shape)
xTrain = xTrain.reshape((nsamples,nx*ny))

xTrain = xTrain.astype('float32')
xTrain /= 255.0

# build the classifier
classifier = svm.SVC(gamma=0.001)
classifier.fit(xTrain, yTrain)

print(xTest.shape)
#print(xTest[0, 187])
nsamples, nx, ny,k = xTest.shape
xTest = xTest.reshape((nsamples,nx*ny))

xTest = xTest.astype('float32')
xTest = xTest/255.0
#print(xTest[0,187])

# made prediction using test data
predict_result = classifier.predict(xTest)
print(len(predict_result))
#result_list = predict_result.tolist()


conf = confusion_matrix(yTest, predict_result)

score = accuracy_score(yTest,predict_result, normalize=False)

import seaborn as sns
import matplotlib.pyplot as plt
plt.close()
plt.title('Confusion matrix of SVM')
sns.heatmap(conf, annot = True)
plt.xlabel('True')
plt.ylabel('False')
plt.savefig('Confusion matrix of SVM')

from sklearn.metrics import classification_report
# plot the first four digits
print(classification_report(yTest, predict_result))
