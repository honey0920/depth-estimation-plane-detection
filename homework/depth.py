#!/usr/bin/env python
# Master's Thesis - Depth Estimation by Convolutional Neural Networks
# Jan Ivanecky; xivane00@stud.fit.vutbr.cz
# Modified by Cheng Han ni for Image Analysis and Understanding course assignment
# chenghn@hust.edu.cn Huazhong University of Science & Technology 
import numpy as np
import cv2
import cv
import os.path
os.environ['GLOG_minloglevel'] = '2' 
import caffe
import scipy.ndimage

WIDTH = 298
HEIGHT = 218
OUT_WIDTH = 74
OUT_HEIGHT = 54
GT_WIDTH = 420
GT_HEIGHT = 320
DIR = 'media/'

def testNet(net, img):	
	net.blobs['X'].data[...] = img	
	net.forward()
	output = net.blobs['depth-refine'].data
	output = np.reshape(output, (1,1,OUT_HEIGHT, OUT_WIDTH))
	return output
	
def loadImage(path, channels, width, height):
	img = caffe.io.load_image(path)
	img = caffe.io.resize(img, (height, width, channels))
	img = np.transpose(img, (2,0,1))
	img = np.reshape(img, (1,channels,height,width))
	return img

def printImage(img, name, channels, width, height):
	params = list()
	params.append(cv.CV_IMWRITE_PNG_COMPRESSION)
	params.append(8)
	imgnp = np.reshape(img, (height,width, channels))
	imgnp = np.array(imgnp * 255, dtype = np.uint8)
	cv2.imwrite(name, imgnp, params)

def eval(out, gt, rawResults):
	linearGT = gt * 10.0
	linearOut = out * 10.0
	rawResults = [x + y for x, y in zip(rawResults, Test(linearOut, linearGT))]
	return rawResults

def ProcessToOutput(depth):
	depth = np.clip(depth, 0.001, 1000)	
	return np.clip(2 * 0.179581 * np.log(depth) + 1, 0, 1)

def get_depth(imagename):		
	caffe.set_mode_cpu()
	netFile = 'model/net_deploy.prototxt'
	modelFile = 'model/model_norm_abs_100k.caffemodel'
	net = caffe.Net(netFile, modelFile, caffe.TEST)
	input_image = cv2.imread(imagename)
	res_input=cv2.resize(input_image,(420,320),interpolation=cv2.INTER_CUBIC)
	input = loadImage(imagename, 3, WIDTH, HEIGHT)
	input *= 255
	input -= 127
	output = testNet(net, input)
	outWidth = OUT_WIDTH
	outHeight = OUT_HEIGHT
	scaleW = float(GT_WIDTH) / float(OUT_WIDTH)
	scaleH = float(GT_HEIGHT) / float(OUT_HEIGHT)
	output = scipy.ndimage.zoom(output, (1,1,scaleH,scaleW), order=3)
	outWidth *= scaleW
	outHeight *= scaleH
	#input += 127
	#input = input / 255.0
	#input = np.transpose(input, (0,2,3,1))
	#input = input[:,:,:,(2,1,0)]
	output = ProcessToOutput(output)
	path1 = DIR+'img.png'
	path2 = DIR+'depth.png'
	cv2.imwrite(path1, res_input)
	printImage(output, path2, 1, int(outWidth), int(outHeight))

