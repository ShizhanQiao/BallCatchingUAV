import numpy as np
import cv2
from numba import njit
from numba.typed import List
import time
import sklearn.cluster as skc  # 密度聚类
from sklearn import metrics   # 评估模型
import matplotlib.pyplot as plt  # 可视化绘图

def cvshow(img):
	cv2.imshow("test",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def EventLoader(file):
	#读入事件数据，并跳过第一行
	return np.loadtxt(file,skiprows=1,delimiter=' ')

@njit()
def count_none_zero(a,i,j,step):
	count = 0
	pointlist = List()
	#构建矩形坐标
	top = 1000
	left = 1000
	right = 0
	bottom = 0 
	for y in range(a.shape[1]):
		for x in range(a.shape[0]):
			if a[x,y] >0:
				count += 1
				#绝对坐标
				location = (i-step+x,j-step+y)
				#左上角
				if location[0] < top:
					top = location[0]
				if location[1] > right:
					right  = location[1]
				if location[0] > bottom:
					bottom = location[0]
				if location[1] < left:
					left = location[1]
				pointlist.append(location)
	return count,pointlist,[(top,left),(bottom,right)]


@njit
def CreateTimeSlidingWindow(eventarray,deltat):
	#创建一个deltat的时间滑动窗口
	t0 = eventarray[0][0]
	for i in range(len(eventarray)):
		if eventarray[i][0] - t0 > deltat:
			return eventarray[0:i]
			break
		eventarray[i][0] -= t0

@njit
def ConstructEventCount(eventarray,w,h):
	#构建记录event轨迹的图片
	I = np.zeros((w,h))
	for i in range(len(eventarray)):
		#格式：I[x,y]=value
		I[int(eventarray[i][1])][int(eventarray[i][2])] += 1

	return I

@njit
def NormalizeTimeStamp(eventarray,I):
	#归一化事件时间
	deltaT = eventarray[-1][0] - eventarray[0][0]
	'''
	sigmaT = sum(list(set([i[0] for i in eventarray])))
	
	for event in eventarray:
		pixel = I[int(event[2])][int(event[1])]
		event[0] = sigmaT/pixel
	meanT =  eventarray.mean(axis = 0)[0]
	for event in eventarray:
		event[0] = (event[0]-meanT)/deltaT
	#实践证明：直接计算rou没办法将时间归一到-1-1的范围内，因此就那样了'''
	#尝试使用C语言风格+njit加速
	array_time_stamp = List()
	sigmaT = 0
	for i in range(len(eventarray)):
		if eventarray[i][0] not in array_time_stamp:
			array_time_stamp.append(eventarray[i][0])
			sigmaT += eventarray[i][0]
	
	#sigmaT = 0.045
	meanT = 0
	count = 0
	T = np.zeros((I.shape[0],I.shape[1]))
	for y in range(I.shape[1]):
		for x in range(I.shape[0]):
			if I[x,y] != 0:
				T[x,y] = sigmaT/I[x,y]
				meanT += T[x,y]
				count += 1

	#计算mean
	meanT = meanT/count

	#生成rou
	P = np.zeros((T.shape[0],T.shape[1]))
	for y in range(T.shape[1]):
		for x in range(T.shape[0]):	
			if T[x,y] != 0:
				P[x,y] = (T[x,y] - meanT)/deltaT

	#return eventarray
	return P+1

@njit
def ThresholdFilter(P,threshold=0.8):
	for y in range(P.shape[1]):
		for x in range(P.shape[0]):
			if P[x,y] >= threshold:
				P[x,y] = 0
			else:
				P[x,y] = -P[x,y]
	return P-1

#@njit
def DBSCAN(P,eps=15,minpts=10):
	pointlist = []
	for y in range(P.shape[1]):
		for x in range(P.shape[0]):
			if P[x,y] > 0:
				pointlist.append([x,y])
	pointlist = np.array(pointlist)
	db = skc.DBSCAN(eps, minpts).fit(pointlist)
	labels = db.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	cluster_list = []
	#找到最大的簇并标在图上
	for i in range(n_clusters_):
		one_cluster = pointlist[labels == i]
		cluster_list.append([len(one_cluster),one_cluster])
	cluster_list.sort(key = lambda x:x[0],reverse = True)
	P = np.zeros((P.shape[0],P.shape[1]))
	for pixel in cluster_list[0][1]:
		P[pixel[0],pixel[1]] = 1

	return P

for i in range(100):
	eventarray = EventLoader("event.txt")
	eventarray = CreateTimeSlidingWindow(eventarray,0.01)
	a = time.time()
	I = ConstructEventCount(eventarray,640,480)
	P = NormalizeTimeStamp(eventarray,I)
	P = ThresholdFilter(P,0.8)
	b = time.time()
	P = DBSCAN(P)
	print(1000*(b-a))
