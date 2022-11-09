from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.urls import reverse

from .models import PicUpload
from .forms import ImageForm

import json
import numpy as np
import cv2
import yaml
import onnxruntime as rt
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
    if image_path != '':
        img = prep_img(image_path)
        inputDetails = model.get_inputs()
        predictions = model.run(None, {inputDetails[0].name: img})
        result = categories[int(np.argmax(predictions))]

    # content = json.simplejson.dumps({'documents': documents, 'image_path': image_path, 'form': form, 'result': result})
    # return HttpResponse(content, content_type='application/json')
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
    model = rt.InferenceSession('../model/model.onnx')
    return model
