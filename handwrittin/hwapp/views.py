from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

from .models import PicUpload
from .forms import ImageForm

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
    return render(request, 'index.html',
                  {'documents': documents, 'image_path': image_path, 'form': form})
