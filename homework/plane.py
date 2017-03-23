#!/usr/bin/env python
#Course assignment of Image Analysis and Understanding-Depth estimation and Plane Detection
#Cheng hanni; chenghn@hust.edu.cn
from skimage.segmentation import slic
from skimage.segmentation import find_boundaries
from skimage.util import img_as_float
import cv2
import numpy as np
import random
import sys
import copy
from PIL import ImageFile  
ImageFile.LOAD_TRUNCATED_IMAGES = True  
DIR = '/home/hanni/test_django/media/'
#get the top-left and thr bottom-right point of an area,the two functions below will be used
#in display_box to draw the bounding box.
def get_tl(pts):
    min_pt = [sys.maxint,sys.maxint]
    for i in range(len(pts)):
        if (pts[i][0] < min_pt[1]):
            min_pt[1] = pts[i][0]
        if (pts[i][1] < min_pt[0]):
            min_pt[0] = pts[i][1]
    return tuple(min_pt)

def get_br(pts):
    max_pt = [0,0]
    for i in range(len(pts)):
        if(pts[i][0] > max_pt[1]):
            max_pt[1] = pts[i][0]
        if(pts[i][1] > max_pt[0]):
            max_pt[0] = pts[i][1]
    return tuple(max_pt)

def get_adjMatrix(clusters,contours):
    """
    Get the adjcent matrix of clusters used for BFS algorithm
    Args:
        clusters: segments from SLIC algorithm(PAMI2012)
        contours: a bool matrix to decide whether a pixel is the edge of clusters(segments) or not
    Returns:
        Matrix of shape (seg_num, seg_num)
    """
    dx = [-1,0,1,0,-1,-1,1,1]
    dy = [0,1,0,-1,1,-1,1,-1]
    seg_num = np.amax(clusters)+1
    adj_matrix = np.zeros((seg_num,seg_num))
    for i in range(clusters.shape[0]):
        for j in range(clusters.shape[1]):
            for k in range(8):
                x = i + dx[k]
                y = j + dy[k]
            if(x>0 and x<clusters.shape[0] and y>0 and y<clusters.shape[1]):
                if(contours[i][j] and  clusters[i][j]!=clusters[x][y]):
                    adj_matrix[clusters[i][j]][clusters[x][y]]=1
                    adj_matrix[clusters[x][y]][clusters[i][j]]=1
    return adj_matrix

def get_clusters(clusters):
    idx_cluster = []
    seg_num = np.amax(clusters)+1
    for i in range(seg_num):
        idx_cluster.append([])
    for i in range(clusters.shape[0]):
        for j in range(clusters.shape[1]):
            idx_cluster[clusters[i][j]].append((i,j))
    return idx_cluster

def get_samples(idx_clusters,idx,sample_num,depth):
    """
    Get the samples of target cluster
    Args:
        idx: the cluster index
        sample_num: number of samples you want to get
        depth: depth map
    Returns:
        sample of 3D points (shape(sample_num,3))
    """
    random.shuffle(idx_clusters[idx])
    samples = idx_clusters[idx][0:sample_num]
    new_samples = []
    for i in range(len(samples)):
        point = [0,0,0]
        point[0] = samples[i][0]
        point[1] = samples[i][1]
        point[2] = depth[point[0]][point[1]]
        new_samples.append(point)
    return np.array(new_samples)

def fit_plane(samples):
    """
    To fit a plane according to a series of 3D points
    Args: 
        samples
    Returns: 
        norm_vec: the normal vector of the fitted plane
    """
    centroid = np.mean(samples, axis=0)
    for i in range(samples.shape[0]):
        samples[i] = samples[i] - centroid
    cov = np.dot(np.transpose(samples),samples)
    U,s,V = np.linalg.svd(cov)
    norm_vec = V[2][:]
    return norm_vec

def is_coplanar(sample1, sample2):
    """
    To decide if two cluster of pixels are coplanar
    Args:
        sample1,sample2
    Returns:
        True or False
    """
    vec1 = fit_plane(sample1)
    vec2 = fit_plane(sample2)
    mag1 = np.sqrt(vec1.dot(vec1))
    mag2 = np.sqrt(vec2.dot(vec2))
    cosine = (vec1.dot(vec2))/(mag1 * mag2)
    if np.fabs(cosine)> 0.97:
        return True
    else:
        return False

def get_plane(idx_clusters,root):
    """
    To get the planar cluster using the root vector 
    Args:
        root: a list that maps the origin cluster with the planar cluster
    Returns:
        plane_cluster
    """
    plane_cluster = copy.copy(idx_clusters)
    for i in range(len(idx_clusters)):
        rootNum = root[i]
        if rootNum!=i:
            plane_cluster[rootNum].extend(plane_cluster[i])
            plane_cluster[i]=[]
    return plane_cluster
    
def display_avg_plane(image,plane_clusters):
    new_image = copy.copy(image)
    colors = []
    for i in range(len(plane_clusters)):
        if plane_clusters[i]!=[]:
            avg_color = np.zeros((3,))
            for j in range(len(plane_clusters[i])):
                x = plane_clusters[i][j][0]
                y = plane_clusters[i][j][1]
                avg_color += np.array(image[x][y])
            avg_color /= len(plane_clusters[i])
            colors.append(avg_color)
        else:
            colors.append(-1)
    for i in range(len(plane_clusters)):
        if plane_clusters[i] !=[]:
            color = colors[i]
            for j in range(len(plane_clusters[i])):
                x = plane_clusters[i][j][0]
                y = plane_clusters[i][j][1]
                new_image[x][y] = color
    cv2.imwrite(DIR+'avg_color.png',new_image)

def display_box(image,plane_clusters):
    new_image = copy.copy(image)
    for i in range(len(plane_clusters)):
        if(len(plane_clusters[i])>2500):
            cv2.rectangle(new_image,get_tl(plane_clusters[i]),get_br(plane_clusters[i]),(0,0,255),1)
    cv2.imwrite(DIR+'box.png',new_image)

def get_planar(image_path,depth_path): 
	image = cv2.imread(image_path)
	depth = cv2.imread(depth_path,0)   
	clusters = slic(image,600,3,10,convert2lab=True)
	contours = find_boundaries(clusters)
	idx_clusters = get_clusters(clusters)
	seg_num = np.amax(clusters)+1
	#initilaize the parameters used in BFS algorihm
	adj_matrix = get_adjMatrix(clusters,contours)
	visited = []
	root = []
	qNode=[]
	sample_num = 30
	for i in range(seg_num):
		visited.append(False)
		root.append(i)
	#Begin BFS
	for i in range(seg_num):
		if(visited[i]==False):
			visited[i] = True
			qNode.append(i)
			while(len(qNode)!=0):
				current = qNode.pop(0)
				for j in range(seg_num):
					if(adj_matrix[current][j]!=0 and visited[j]==False):
						visited[j] = True
						qNode.append(j)
						sample1 = get_samples(idx_clusters,current,sample_num,depth)
						sample2 = get_samples(idx_clusters,j,sample_num,depth)
						if(is_coplanar(sample1,sample2)):
							root[j] = root[current]
	#end BFS
	plane_clusters = get_plane(idx_clusters,root)
	display_avg_plane(image,plane_clusters)  #display the average color of each clusters(ICRA 2009)
	display_box(image,plane_clusters) #display bounding boxes of planar area
	#cv2.waitKey(0)
