#%%

from keras.preprocessing import sequence
from keras.datasets import imdb
from matplotlib import pyplot as plt
import pandas as pd


max_features = 20000



print('Loading data...')

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
#标签转换为独热码
y_train, y_test = pd.get_dummies(y_train),pd.get_dummies(y_test)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')



#%%数据归一化处理

maxlen = 64


print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)

#%%

batch_size = 5
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.layers import *
from Attention_keras import Attention,Position_Embedding


S_inputs = Input(shape=(None,), dtype='int32')

embeddings = Embedding(max_features, 128)(S_inputs)
embeddings = Position_Embedding()(embeddings) #增加Position_Embedding能轻微提高准确率

O_seq = Attention(8,16)([embeddings,embeddings,embeddings])

O_seq = GlobalAveragePooling1D()(O_seq)

O_seq = Dropout(0.5)(O_seq)

outputs = Dense(2, activation='softmax')(O_seq)


model = Model(inputs=S_inputs, outputs=outputs)
# try using different optimizers and different optimizer configs
opt = Adam(lr=0.0005)
loss = 'categorical_crossentropy'
model.compile(loss=loss,

             optimizer=opt,

             metrics=['accuracy'])

print(model.summary())
#%%
print('Train...')

model.fit(x_train, y_train,

         batch_size=batch_size,

         epochs=2,

         validation_data=(x_test, y_test))



model.save("imdb.h5")