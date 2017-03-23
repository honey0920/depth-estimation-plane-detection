from django.shortcuts import render ,render_to_response
from django import forms
from django.template.loader import get_template
from django.http import HttpResponse,Http404
import datetime
from django import template
from PIL import Image
from depth import get_depth
from plane import get_planar
# Create your views here.

def demo(request):
	t =  get_template('demo.html')
	html = t.render(template.Context())
	return HttpResponse(html)

def get_pic(request):
	if request.method=='POST':
		try:
			image = request.FILES['image']
			img = Image.open(image)
			filepath='media/origin.png'
			img.save(filepath)
			get_depth(filepath)
			get_planar('media/img.png','media/depth.png')
			return render_to_response('show_pic.html',{'image':'/media/test.png'})
		except Exception,e:
			return HttpResponse(e)
		
	
