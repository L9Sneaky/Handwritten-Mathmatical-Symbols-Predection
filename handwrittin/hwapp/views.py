from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

from .models import PicUpload
from .forms import ImageForm

import numpy as np
import cv2
import yaml
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Create your views here.


def index(request):
    image_path = ''
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = PicUpload(imagefile=request.FILES['imagefile'])
            newdoc.save()
            return HttpResponseRedirect(reverse('index'))
    else:
        form = ImageForm()

    documents = PicUpload.objects.all()
    for document in documents:
        image_path = document.imagefile.name
        document.delete()

    request.session['image_path'] = image_path

    categories = get_category()
    model = get_model()
    result = ''
    if image_path is not '':
        img = prep_img(image_path)
        # result = img.shape
        # result = model.summary()
        predictions = model.predict(img)
        result = categories[int(np.argmax(predictions))]
    return render(request, 'index.html',
                  {'documents': documents, 'image_path': image_path, 'form': form, 'result': result})


def get_category():
    categories = []
    with open("../model/categories.yaml", 'r') as stream:
        categories = yaml.safe_load(stream)
    return categories


def prep_img(img_path):
    IMG_SIZE = 64
    x = cv2.imread(img_path)
    x = x / 255.0
    x = cv2.resize(x, (IMG_SIZE, IMG_SIZE))
    x = np.expand_dims(x, axis=0)
    return x


def get_model():
    model = load_model('../model/model.h5')
    model.load_weights('../model/modelW.h5')
    return model
