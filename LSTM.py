import tensorflow
import sklearn
import pandas as pd
import os
def excel_to_pd(path,columns=['Αριθμός Κλήρωσης', 'Ημ/νία Κλήρωσης', 'Ώρα Κλήρωσης', '1ος ', '2ος ',
       '3ος ', '4ος ', '5ος ', '6ος ', '7ος ', '8ος ', '9ος ', '10ος ',
       '11ος ', '12ος ', '13ος ', '14ος ', '15ος ', '16ος ', '17ος ', '18ος ',
       '19ος ', '20ος ']):
    x = pd.read_excel(path)[columns]
    return x
paths= [ "data/raw/" + x for x in sorted( os.listdir("data/raw"))]
dfs = [excel_to_pd(path) for path in paths]
kino = pd.concat(dfs,axis=0)
del dfs
map_names = {'Αριθμός Κλήρωσης':'ID', 'Ημ/νία Κλήρωσης':'Date', 'Ώρα Κλήρωσης':'Time', '1ος ':'nr1', '2ος ':'nr2',
       '3ος ':'nr3', '4ος ':'nr4', '5ος ':'nr5', '6ος ':'nr6', '7ος ':'nr7', '8ος ':'nr8', '9ος ':'nr9', '10ος ':'nr10',
       '11ος ':'nr11', '12ος ':'nr12', '13ος ':'nr13', '14ος ':'nr14', '15ος ':'nr15', '16ος ':'nr16', '17ος ':'nr17', '18ος ':'nr18',
       '19ος ':'nr19', '20ος ':'nr20'}
kino.rename(columns=map_names,inplace=True)
type(kino.Date)
kino["datetime"] = pd.to_datetime(kino['Date'] + ' '+ kino['Time'])
kino.sort_values(by='datetime',inplace=True)
kino.dropna(inplace=True)
#nr_cols = ['nr' + str(x) for x in list(range(1,21))]
nr_cols = [ x for x in kino.columns if 'nr' in x]
kino[nr_cols] = kino[nr_cols].astype(int)
for num in range(1,81):
    kino["num"+str(num)] = (kino[nr_cols].astype(int) == num).sum(axis=1)
kino.drop(nr_cols,axis=1,inplace=True)
num_cols = [ x for x in kino.columns if 'num' in x]
kino.iloc[-1,]
target = kino[num_cols].shift(-1)
kino = kino[:-1]
target = target[:-1]
import  tensorflow as tf
import  tensorflow.python.keras  as keras
import tensorflow.python.keras.losses as losses
lookback=5
slices=[]
slices_target = target.iloc[5:target.shape[0]]
for i in range(lookback,kino.shape[0]):
    slices.append( kino[(i-lookback):i][num_cols])
import numpy as np
tensor = np.stack(slices)
tensor = tensor.astype("float32")
model_lstm = keras.Sequential()
model_lstm.add(keras.layers.LSTM(80,input_shape=( 5,80)))
model_lstm.add(keras.layers.Dense(80,activation="sigmoid"))
model_lstm.compile(optimizer="SGD",loss=keras.losses.binary_crossentropy, metrics=["accuracy"])
train_tensor = tensor[:-200]
test_tensor = tensor[-200:]
train_target = slices_target[:-200]
test_target = slices_target[-200:]
model_lstm.fit(train_tensor,train_target.values,batch_size=128)
train_kino = kino[ kino.datetime < '2019-12-01 00:00:00']
test_kino = kino[kino.datetime > '2019-12-01 00:00:00']
train_y = target[ kino.datetime < '2019-12-01 00:00:00']
test_y = target[ kino.datetime > '2019-12-01 00:00:00']
model = keras.Sequential()
model.add(keras.layers.Dense(40,activation="relu",input_shape=(80,)))
model.add(keras.layers.Dense(80,activation="softmax"))
import numpy as np
model.compile(loss=keras.losses.binary_crossentropy,optimizer="SGD",metrics=["accuracy"])
history=model.fit(train_kino[num_cols].values.astype("float32"),train_y.values,epochs=10,validation_split=0.2)
from matplotlib import pyplot as plt
history.history.keys()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
predictions = model.predict(test_kino[num_cols])
pd.set_option('display.expand_frame_repr', False)
test_y[0:3]
test_kino["num1"]
predictions[:,0]
df = pd.DataFrame({ "actual": test_kino["num1"],"prediction":predictions[:,0]}  )
df_sorted = df.sort_values(by="prediction",ascending=False)
df_sorted[df_sorted.actual==1].describe()
df_sorted[df_sorted.actual==0].describe()
