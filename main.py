#!/usr/bin/env python
# coding: utf-8

# In[12]:


from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import ipywidgets as widgets
from ipywidgets import interact
import imageio


# In[3]:


model = YOLO("/Users/daryarozhnovskaya/DIPLOM2.0/runs/segment/train2/weights/best.pt")


# In[4]:


img = cv2.imread("/Users/daryarozhnovskaya/Desktop/Датасет дерматоскопия/NE - пигментный невус/0-0-660273.#1.tif")

result = model(img, imgsz=640, iou=0.4, conf=0.36, verbose=True)


# In[7]:


result


# In[4]:


plt.imshow(result[0].masks.data[0].cpu().numpy(), 'gray')


# In[8]:


# Получение классов и имен классов
classes = result[0].boxes.cls.cpu().numpy()
class_names = result[0].names

# Получение бинарных масок и их количество
masks = result[0].masks.data  # Формат: [число масок, высота, ширина]
num_masks = masks.shape[0]

# Определение случайных цветов и прозрачности для каждой маски
colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_masks)]  # Случайные цвета

# Создание изображения для отображения масок
mask_overlay = np.zeros_like(img)

labeled_image = img.copy()


# Добавление подписей к маскам
for i in range(num_masks):
    color = colors[i]  # Случайный цвет
    mask = masks[i].cpu()

    # Изменение размера маски до размеров исходного изображения с использованием метода ближайших соседей
    mask_resized = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Получение класса для текущей маски
    class_index = int(classes[i])
    class_name = class_names[class_index]

    # Добавление подписи к маске
    mask_contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(labeled_image, mask_contours, -1, color, 3)
    cv2.putText(labeled_image, class_name, (int(mask_contours[0][:, 0, 0].mean()), int(mask_contours[0][:, 0, 1].mean())),
                cv2.FONT_HERSHEY_SIMPLEX, 3, color, 6)

# Отобразите итоговое изображение с наложенными масками и подписями
plt.figure(figsize=(8, 8), dpi=150)
labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
plt.imshow(labeled_image)
imageio.imwrite('/Users/daryarozhnovskaya/Desktop/Новая папка 3/Modified/image.png', labeled_image.astype(np.uint8))
plt.axis('off')
plt.show()


# In[ ]:


results = model.predict("/Users/daryarozhnovskaya/Downloads/08d214276fb48a6ce87c17e2fbff3b14.jpg", save=True, imgsz=640, conf=0.5)


# In[ ]:


boxes = results[0].boxes.xywh.cpu()
for box in boxes:
    x, y, w, h = box
    print(f"Width of Box: {w}, Height of Box: {h}")


# In[9]:


image = cv2.imread('/Users/daryarozhnovskaya/DIPLOM2.0/YOLO_dataset/train/images/ISIC_0024307.JPG')
 

def interactive_plot(alpha, iou, conf, imgsz):
    np.random.seed(42)
    # Инференс с использованием модели YOLOv5
    results = model(image, imgsz=imgsz, iou=iou, conf=conf, verbose=False);

   # Получение бинарных масок и их количество
    masks = results[0].masks.data  # Формат: [число масок, высота, ширина]
    num_masks = masks.shape[0]

    # Определение случайных цветов и прозрачности для каждой маски
    colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_masks)]  # Случайные цвета

    # Создание изображения для отображения масок
    mask_overlay = np.zeros_like(image)

    # Наложение масок на изображение
    for i in range(num_masks):
        color = colors[i]  # Случайный цвет
        mask = masks[i].cpu()

        # Изменение размера маски до размеров исходного изображения с использованием метода ближайших соседей
        mask_resized = cv2.resize(np.array(mask), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        #print(mask.shape, img.shape, mask_resized.shape)

        # Создание маски с цветом и прозрачностью
        color_mask = np.zeros_like(image)
        color_mask[mask_resized > 0] = color
        mask_overlay = cv2.addWeighted(mask_overlay, 1, color_mask, alpha, 0)

    # Объединение исходного изображения и масок
    result_image = cv2.addWeighted(image, 1, mask_overlay, 1, 0)

    # Отобразите итоговое изображение с наложенными масками
    plt.figure(figsize=(8, 8), dpi=150)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    plt.imshow(result_image)
    plt.axis('off')
    plt.show()

# Создайте виджеты для изменения параметров
alpha_slider = widgets.FloatSlider(value=0.20, min=0.0, max=1.0, step=0.05, description='Alpha')
iou_slider = widgets.FloatSlider(value=0.65, min=0.0, max=1.0, step=0.05, description='IOU')
conf_slider = widgets.FloatSlider(value=0.15, min=0.0, max=1.0, step=0.05, description='Confidence')
imgsz_slider = widgets.IntSlider(value=608, min=32, max=2000, step=32, description='imgsz')

# Используйте interact для связи виджетов с функцией
interact(interactive_plot, alpha=alpha_slider, iou=iou_slider, conf=conf_slider, imgsz=imgsz_slider);


# In[1]:


import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import ipywidgets as widgets
from ipywidgets import interact
from PyQt5.QtWidgets import (
    QApplication, QWidget, 
    QFileDialog, # Диалог открытия файлов (и папок)
    QLabel, QPushButton, QListWidget,
    QHBoxLayout, QVBoxLayout
)
from PyQt5.QtCore import Qt # нужна константа Qt.KeepAspectRatio для изменения размеров с сохранением пропорций
from PyQt5.QtGui import QPixmap # оптимизированная для показа на экране картинка

from PIL import Image
import imageio

#from PIL.ImageQt import ImageQt # для перевода графики из Pillow в Qt  

app = QApplication([])
win = QWidget()        
win.resize(700, 500)  
win.setWindowTitle('Дипломная работа')
lb_image = QLabel("Картинка")
labels = QLabel("NE - невус, MM - меланома, BCC - базальная клеточная карцинома,\nSCC - плоскоклеточная карцинома, DF - дерматофиброма, HE - гиперкератоз кожи,\nAK - актинический кератоз")
btn_dir = QPushButton("Загрузить")
lw_files = QListWidget()


btn_back = QPushButton("Сброс")
btn_seg = QPushButton("Провести сегментацию")

row = QHBoxLayout()          # Основная строка 
col1 = QVBoxLayout()         # делится на два столбца
col2 = QVBoxLayout()
col1.addWidget(btn_dir)      # в первом - кнопка выбора директории
col1.addWidget(lw_files)     # и список файлов
col2.addWidget(lb_image, 95) # вo втором - картинка
col2.addWidget(labels) # вo втором - картинка

row_tools = QHBoxLayout()    # и строка кнопок

row_tools.addWidget(btn_back)
row_tools.addWidget(btn_seg)
col2.addLayout(row_tools)

row.addLayout(col1, 20)
row.addLayout(col2, 80)
win.setLayout(row)

win.show()
model = YOLO("/Users/daryarozhnovskaya/DIPLOM2.0/runs/segment/train2/weights/best.pt")

workdir = ''

def filter(files, extensions):
    result = []
    for filename in files:
        for ext in extensions:
            if filename.endswith(ext):
                result.append(filename)
    return result

def chooseWorkdir():
    global workdir
    workdir = QFileDialog.getExistingDirectory()

def showFilenamesList():
    extensions = ['.jpg','.jpeg', '.png', '.gif', '.bmp',".tif"]
    chooseWorkdir()
    filenames = filter(os.listdir(workdir), extensions)

    lw_files.clear()
    for filename in filenames:
        lw_files.addItem(filename)

btn_dir.clicked.connect(showFilenamesList)


class ImageProcessor():
    def __init__(self):
        self.image = None
        self.dir = None
        self.filename = None
        self.save_dir = "Modified/"
        self.image_path = None

    def loadImage(self, dir, filename):
        ''' при загрузке запоминаем путь и имя файла '''
        self.dir = dir
        self.filename = filename
        self.image_path = os.path.join(dir, filename)
        self.image = Image.open(self.image_path)

    def seg(self):
        model = YOLO("/Users/daryarozhnovskaya/DIPLOM2.0/runs/segment/train2/weights/best.pt")
        self.loadImage(self.dir, self.filename)
        self.image = cv2.imread(self.image_path)
        result = model(self.image, imgsz=640, iou=0.8, conf=0.25, verbose=True)
        # Получение классов и имен классов
        classes = result[0].boxes.cls.cpu().numpy()
        class_names = result[0].names

        # Получение бинарных масок и их количество
        masks = result[0].masks.data  # Формат: [число масок, высота, ширина]
        num_masks = masks.shape[0]

        # Определение случайных цветов и прозрачности для каждой маски
        colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_masks)]  # Случайные цвета

        # Создание изображения для отображения масок
        mask_overlay = np.zeros_like(self.image)

        labeled_image = self.image.copy()


        # Добавление подписей к маскам
        for i in range(num_masks):
            color = colors[i]  # Случайный цвет
            mask = masks[i].cpu()

            # Изменение размера маски до размеров исходного изображения с использованием метода ближайших соседей
            mask_resized = cv2.resize(np.array(mask), (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Получение класса для текущей маски
            class_index = int(classes[i])
            class_name = class_names[class_index]

            # Добавление подписи к маске
            mask_contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(labeled_image, mask_contours, -1, color, 3)
            cv2.putText(labeled_image, class_name, (int(mask_contours[0][:, 0, 0].mean()), int(mask_contours[0][:, 0, 1].mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 6)

        # Отобразите итоговое изображение с наложенными масками и подписями
        labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        imageio.imwrite(f'/Users/daryarozhnovskaya/Desktop/iso/Modified/{self.filename}', labeled_image.astype(np.uint8))
        self.image_path = os.path.join(self.dir, self.save_dir, self.filename)
        self.showImage(f'/Users/daryarozhnovskaya/Desktop/iso/Modified/{self.filename}')
        

    def saveImage(self):
        ''' сохраняет копию файла в подпапке '''
        path = os.path.join(self.dir, self.save_dir)
        if not(os.path.exists(path) or os.path.isdir(path)):
            os.mkdir(path)
        self.image_path = os.path.join(path, self.filename)
        self.image.save(self.image_path)

    def showImage(self, path):
        lb_image.hide()
        pixmapimage = QPixmap(path)
        w, h = lb_image.width(), lb_image.height()
        pixmapimage = pixmapimage.scaled(w, h, Qt.KeepAspectRatio)
        lb_image.setPixmap(pixmapimage)
        lb_image.show()

def showChosenImage():
    if lw_files.currentRow() >= 0:
        filename = lw_files.currentItem().text()
        workimage.loadImage(workdir, filename)
        image_path = os.path.join(workimage.dir, workimage.filename)
        workimage.showImage(image_path)

workimage = ImageProcessor() #текущая рабочая картинка для работы
lw_files.currentRowChanged.connect(showChosenImage)

btn_seg.clicked.connect(workimage.seg)



app.exec()


# In[31]:


import mlflow


# In[10]:


import pandas as pd
import matplotlib.pyplot as plt


# In[244]:


df = pd.read_csv("/Users/daryarozhnovskaya/DIPLOM2.0/runs/segment/train2/results.csv")


# In[224]:


df.info()


# In[208]:


df.columns


# In[214]:


df


# In[217]:


model.val()


# In[218]:


boxes = results[0].boxes.xywh.cpu()
for box in boxes:
    x, y, w, h = box
    print(f"Width of Box: {w}, Height of Box: {h}")


# In[221]:


from threading import Thread


def predict(image_path):
    """Predicts objects in an image using a preloaded YOLO model, take path string to image as argument."""
    results = model.predict(image_path)
    # Process results


# Starting threads that share the same model instance
Thread(target=predict, args=("/Users/daryarozhnovskaya/DIPLOM2.0/YOLO_dataset/train/images/ISIC_0024308.JPG",)).start()
Thread(target=predict, args=("/Users/daryarozhnovskaya/DIPLOM2.0/YOLO_dataset/train/images/ISIC_0024318.JPG",)).start()


# In[18]:


import datetime
import shutil
from pathlib import Path
from collections import Counter

import yaml
import pandas as pd
from sklearn.model_selection import KFold


# In[ ]:


dataset_path = Path('./Fruit-detection') # replace with 'path/to/dataset' for your custom data
labels = sorted(dataset_path.rglob("*labels/*.txt")) # all data in 'labels'


# In[8]:


model_val = YOLO("/Users/daryarozhnovskaya/DIPLOM2.0/runs/segment/train2/weights/best.pt")


# In[19]:


data = pd.read_csv("/Users/daryarozhnovskaya/DIPLOM2.0/runs/segment/train2/results.csv")


# In[21]:


data.info()


# In[18]:


data.columns


# In[27]:


# Данные для графика 1
x1 = data["       metrics/mAP50(B)"]
x2 = data["       metrics/mAP50(M)"]


# Построение обоих графиков на одном полотне
plt.figure(figsize=(10, 5))  # Установка размера графика

plt.plot(x1,  color='b', label='Тренировочная выборка')  # построение графика 1
plt.plot(x2, color='r', label='Валидационная выборка')  # построение графика 2

plt.title('Средняя точность предсказания')  # заголовок графика
plt.xlabel('Количество эпох')  # подпись оси X
plt.ylabel('mAP50')  # подпись оси Y
plt.legend()  # добавление легенды
plt.xticks (ticks=[0,5,10,15,20], labels=[0,5,10,15,20]) 

plt.show()  # Отображение графиков


# In[28]:


# Данные для графика 1
x1 = data["         train/seg_loss"]
x2 = data["           val/seg_loss"]


# Построение обоих графиков на одном полотне
plt.figure(figsize=(10, 5))  # Установка размера графика

plt.plot(x1,  color='b', label='Тренировочная выборка')  # построение графика 1
plt.plot(x2, color='r', label='Валидационная выборка')  # построение графика 2

plt.title('Функция потерь')  # заголовок графика
plt.xlabel('Количество эпох')  # подпись оси X
plt.ylabel('loss function')  # подпись оси Y
plt.legend()  # добавление легенды
plt.xticks (ticks=[0,5,10,15,20], labels=[0,5,10,15,20]) 

plt.show()  # Отображение графиков


# In[29]:


# Данные для графика 1
x1 = data["      metrics/recall(B)"]
x2 = data["      metrics/recall(M)"]


# Построение обоих графиков на одном полотне
plt.figure(figsize=(10, 5))  # Установка размера графика

plt.plot(x1,  color='b', label='Тренировочная выборка')  # построение графика 1
plt.plot(x2, color='r', label='Валидационная выборка')  # построение графика 2

plt.title('Доля верно найденных моделью объектов')  # заголовок графика
plt.xlabel('Количество эпох')  # подпись оси X
plt.ylabel('recall')  # подпись оси Y
plt.legend()  # добавление легенды
plt.xticks (ticks=[0,5,10,15,20], labels=[0,5,10,15,20]) 

plt.show()  # Отображение графиков


# In[30]:


# Данные для графика 1
x1 = data["   metrics/precision(B)"]
x2 = data["   metrics/precision(M)"]


# Построение обоих графиков на одном полотне
plt.figure(figsize=(10, 5))  # Установка размера графика

plt.plot(x1,  color='b', label='Тренировочная выборка')  # построение графика 1
plt.plot(x2, color='r', label='Валидационная выборка')  # построение графика 2

plt.title('Доля верно предсказанных моделью объектов')  # заголовок графика
plt.xlabel('Количество эпох')  # подпись оси X
plt.ylabel('precision')  # подпись оси Y
plt.legend()  # добавление легенды
plt.xticks (ticks=[0,5,10,15,20], labels=[0,5,10,15,20]) 

plt.show()  # Отображение графиков


# In[ ]:




