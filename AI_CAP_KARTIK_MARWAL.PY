import numpy as np
import pandas as pd
import keras
import cv2
from matplotlib import pyplot as plt
import os
import random
from PIL import Image
from keras.utils import plot_model





df = pd.read_csv("train.csv")
base_path ="D:\\my projects\\PYTHON\\images"
# print (df)
samples = 20000
df=df.loc[df["id"].str.startswith('00',na=False), :]
num_classes = len(df["landmark_id"].unique())
num_data = len(df)

print (num_classes)
print (num_data)

data = pd.DataFrame(df["landmark_id"].value_counts())
data.reset_index(inplace=True)
data.columns = ['landmark_id','count']
print(data.head())
print(data.tail())
data.columns = ['landmark_id','count']
print(data['count'].describe())
plt.hist(data['count'],100,range=(0,944),label= 'test')
plt.show()
print(data['count'].between(0,5).sum())
print(data['count'].between(5,10).sum())
plt.hist(df["landmark_id"], bins=len(df["landmark_id"].unique()))
plt.show()

#training of model
from sklearn.preprocessing import LabelEncoder
lencoder = LabelEncoder()
lencoder.fit(df['landmark_id'])
print(df.head())
def encode_label(lbl):
    return lencoder.transform(lbl)
def decode_label(lbl):
    return lencoder.inverse_transform(lbl)
def get_image_from_number(num,df):
    fname, label = df.iloc[num][['id', 'landmark_id']]

    fname=fname +'.jpg'
    f1 = fname[0]
    f2 = fname[1]
    f3 = fname[2]
    path = os.path.join(f1,f2,f3,fname)
    im = cv2.imread(os.path.join(base_path ,path))
    if im is None:
        print(f"Warning: Image at {path} could not be loaded.")
    return im,label

print("4 sample images from random classes")
fig = plt.figure(figsize=(16,16))
for i in range(1,5):
    ri = random.choices(os.listdir(base_path),k=3)
    folder = base_path +'/' + ri[0] +'/' + ri[1] +'/' + ri[2]
    random_img =random.choice(os.listdir(folder))
    img = np.array(Image.open(folder+'/'+random_img))
    fig.add_subplot(1,4,i)
    plt.imshow(img)
    plt.axis('off')
plt.show()    


from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras import Sequential
# tf.compat.v1.disable_eager_execution()

# Parameters
learning_rate = 0.0001
decay_speed = 1e-6
momentum = 0.09
loss_function = "sparse_categorical_crossentropy"
source_model = VGG19(weights=None)
drop_layer = Dropout(0.5)
drop_layer2 = Dropout(0.5)

model = Sequential()
for layer in source_model.layers[:-1]:
    if layer == source_model.layers[-25]:
        model.add(BatchNormalization())
    model.add(layer)
model.add(Dense(num_classes,activation = "softmax"))
print(model.summary())


# Compiling the model
optim1 = keras.optimizers.RMSprop(learning_rate= learning_rate)
model.compile(optimizer = optim1, loss = loss_function, metrics = ["accuracy"])

def image_reshape(im, target_size):
    return cv2.resize(im,target_size)
def get_batch(dataframe,start,batch_size):
    image_array = []
    label_array = []

    end_img = start + batch_size
    if(end_img) >len(dataframe):
        end_img = len(dataframe)
    
    for idx in range(start,end_img):
        n= idx
        im,label = get_image_from_number(n,dataframe)    
        im = image_reshape(im, (224, 224)).astype(np.float32) / 255.0
        image_array.append(im)
        label_array.append(label)

    label_array = encode_label(label_array)
    return np.array(image_array), np.array(label_array)

batch_size= 16
epoch_shuffle = True
weight_classes = True
epochs = 1

#split
train,val = np.split(df.sample(frac=1),[int(0.8*len(df))])
print(len(train))
print(len(val))

for e in range(epochs):
    print("Epoch :" + str(e+1) + "/" + str(epochs))
    if epoch_shuffle:
        train = train.sample(frac=1)
    for it in range(int(np.ceil(len(train)/batch_size))):
        x_train, y_train =get_batch(train, it*batch_size, batch_size)

        model.train_on_batch(x_train, y_train)

print(model.save("Model"))






# # Visualize the model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# # Display the model plot
# img = Image.open('model_plot.png')
# plt.figure(figsize=(12, 12))
# plt.imshow(img)
# plt.axis('off')
# plt.show()


#test
batch_size = 16

errors=0 ;
good_pred = []
bad_pred = []

for it in range(int(np.ceil(len(val)/batch_size))):
    x_val, y_val =get_batch(val, it*batch_size, batch_size)

    result = model.predict_function(x_val)
    cla = np.argmax(result,axis=1)
    for idx,res in enumerate(result):
        if cla[idx] != y_val[idx]:
            errors = errors+1
            bad_pred.append([batch_size*it +idx],cla[idx], res[cla[idx]])
        else:
            good_pred.append([batch_size*it +idx],cla[idx], res[cla[idx]])    

for i in range(1,6):
    n= int(good_pred[0])
    img, lbl =get_image_from_number(n,val)
    img = cv2. cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)           