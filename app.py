import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Model
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Momentum
from mindspore import load_checkpoint, load_param_into_net
import numpy as np
import cv2
from mindspore import Tensor
from mindspore.common import dtype as mstype
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 

class_map = {'Aloevera': 0,
 'Amla': 1,
 'Amruthaballi': 2,
 'Arali': 3,
 'Astma_weed': 4,
 'Badipala': 5,
 'Balloon_Vine': 6,
 'Bamboo': 7,
 'Beans': 8,
 'Betel': 9,
 'Bhrami': 10,
 'Bringaraja': 11,
 'Caricature': 12,
 'Castor': 13,
 'Catharanthus': 14,
 'Chakte': 15,
 'Chilly': 16,
 'Citron lime (herelikai)': 17,
 'Coffee': 18,
 'Common rue(naagdalli)': 19,
 'Coriender': 20,
 'Curry': 21,
 'Doddpathre': 22,
 'Drumstick': 23,
 'Ekka': 24,
 'Eucalyptus': 25,
 'Ganigale': 26,
 'Ganike': 27,
 'Gasagase': 28,
 'Ginger': 29,
 'Globe Amarnath': 30,
 'Guava': 31,
 'Henna': 32,
 'Hibiscus': 33,
 'Honge': 34,
 'Insulin': 35,
 'Jackfruit': 36,
 'Jasmine': 37,
 'Kambajala': 38,
 'Kasambruga': 39,
 'Kohlrabi': 40,
 'Lantana': 41,
 'Lemon': 42,
 'Lemongrass': 43,
 'Malabar_Nut': 44,
 'Malabar_Spinach': 45,
 'Mango': 46,
 'Marigold': 47,
 'Mint': 48,
 'Neem': 49,
 'Nelavembu': 50,
 'Nerale': 51,
 'Nooni': 52,
 'Onion': 53,
 'Padri': 54,
 'Palak(Spinach)': 55,
 'Papaya': 56,
 'Parijatha': 57,
 'Pea': 58,
 'Pepper': 59,
 'Pomoegranate': 60,
 'Pumpkin': 61,
 'Raddish': 62,
 'Rose': 63,
 'Sampige': 64,
 'Sapota': 65,
 'Seethaashoka': 66,
 'Seethapala': 67,
 'Spinach1': 68,
 'Tamarind': 69,
 'Taro': 70,
 'Tecoma': 71,
 'Thumbe': 72,
 'Tomato': 73,
 'Tulsi': 74,
 'Turmeric': 75,
 'ashoka': 76,
 'camphor': 77,
 'kamakasturi': 78,
 'kepala': 79}

class AlexNet(nn.Cell):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, 5, pad_mode='same', group=2)
        self.conv3 = nn.Conv2d(256, 384, 3, pad_mode='same')
        self.conv4 = nn.Conv2d(384, 384, 3, pad_mode='same', group=2)
        self.conv5 = nn.Conv2d(384, 256, 3, pad_mode='same', group=2)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Dense(256*6*6, 4096)
        self.dropout = nn.Dropout(p = 0.5)
        self.dense2 = nn.Dense(4096, 4096)
        self.dense3 = nn.Dense(4096, num_classes)

    def construct(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool2d(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool2d(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.dense1(x)))
        x = self.dropout(self.relu(self.dense2(x)))
        x = self.dense3(x)
        return x

def get_name(id):
    return list(class_map.keys())[list(class_map.values()).index(id)]

net = AlexNet(num_classes=80) 
loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
optimizer = Momentum(net.trainable_params(), learning_rate=0.001, momentum=0.9)
param_dict = load_checkpoint("95.ckpt")
load_param_into_net(net, param_dict)
model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={"accuracy"})


descriptions = open("json.json", 'r').read()
descriptions = json.loads(descriptions)

app = FastAPI() 

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

@app.get("/")
async def root():
    return {"message": "Hello from Fastapi"} 


@app.get("/ping", tags=["Test"])
async def ping():
    return 'pong'

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, (227, 227))
    image_tensor = Tensor(np.array(resized_image, dtype=np.float32).transpose(2, 0, 1))
    image_tensor = image_tensor.expand_dims(0)

    predictions = model.predict(image_tensor)
    predicted_class = np.argmax(predictions.asnumpy(), axis=1)
    class_name = get_name(predicted_class[0])

    return JSONResponse(content={"Predicted Class": class_name})

@app.post("/describe/")
async def describe(plant: str = Query(..., description="The name of the plant to retrieve")):
    print(plant)
    return JSONResponse(content = {plant : descriptions[plant]})

