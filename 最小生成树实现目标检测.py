import numpy as np 
import cv2
import pandas as pd
import random
import datetime
import sys
import copy
#思路：随机选100个点，计算其距离矩阵，然后用最小生成树，规定边长不能大于某个阈值
def cvshow(img,time=1):
	cv2.imshow("test",img)
	cv2.waitKey(time)

def discal(point1,point2):
	return np.sqrt(int((point1[0]-point2[0])**2+(point1[1]-point2[1])**2))

def readCsv(usecolname,path,sep="	",ifreverse=False,iftranspose=True,iflist=False):
	'''readCsv(使用列的名称：列表,文件路径,分隔符="	",是否要反转数据=False)：读取csv文件'''
	df = pd.read_csv(path,sep=sep)
	try:
		series = df[usecolname]
	except KeyError:
		raise KeyError("请检查您所用的列或分隔符是否正确")
	if iftranspose:
		series = np.array(series).T
	else:
		series = np.array(series)
	value_list = []
	if not iflist:
		for i in range(len(series)):
			if ifreverse:
				value_list.append(np.array(series[i][::-1]))
			else:
				value_list.append(np.array(series[i]))
	else:
		for i in range(len(series)):
			if ifreverse:
				value_list.extend(list(series[i][::-1]))
			else:
				value_list.extend(list(series[i]))
	return tuple(value_list)

def event2Point(selected):
	points = []
	for each in selected:
		points.append([int(each[0]),int(each[1])])
	points = np.array(points).T
	return points

def distanceGenerate(eachclass):
	n = len(eachclass)
	A = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			if i==j:
				A[i,j]=0
			else:
				A[i,j]=100000
	#对每一类分别计算距离矩阵，并生成最小生成树，计算总距离，注意一级点距离无穷大
	for point1 in range(len(eachclass)):
		for point2 in range(len(eachclass)):
			A[point1,point2] = discal(eachclass[point1],eachclass[point2])
	#使对称
	for j in range(n):
		for j in range(j):
			A[j,i]=A[i,j]

	return A

def Kruskal(distance,VpositionList,threshold):
	#省略不必须的东西
	edge_list = []
	boundingbox = [] #该列表用于记录边界框
	templistx = []
	templisty = []
	#将边全部取出，放到一个列表中，形状为[(连接点),权值]
	for i in range(distance.shape[0]):
		for j in range(i,distance.shape[1]):
			if i != j:
				edge_list.append([(i,j),distance[i,j]])

	#排序，权值从小到大
	edge_list.sort(key = lambda x:x[1], reverse = False)
	#用可视化技术展示算法过程
	'''
	size = (720,1080)
	canvas = np.zeros((size[1],size[0]),dtype=np.uint8)
	canvas.fill(255)

	#随机选n个点，并渲染文字
	font = cv2.FONT_HERSHEY_SIMPLEX
	'''
	num_of_point = distance.shape[0]

	#加入位置列表，方便画线
	location_list = []
	for i in range(num_of_point):
		locx = VpositionList[0][i]
		locy = VpositionList[1][i]
		location_list.append((locx, locy)) #这里加i是为了索引Vpositionlist
		#cv2.putText(canvas,str(i), (locx, locy), font, 0.1, (0,0,0), 1)

	#换思路：聚类算法
	list_path = []
	edge_class = []
	pahe = []
	total_weight = 0

	for edge in edge_list:
		#划线edge[0][0]表示起点编号，edge[0][1]表示终点编号
		begin,end = edge[0][0],edge[0][1]
		#用于记录状态变量
		flag = -1
		if edge_class:	
			for classes in edge_class:
				if begin in classes or end in classes:
					#不符合新类条件，退出
					flag = 3
					break
			if flag == -1:
				#均不属于任意一类：创建新的类
				edge_class.append([begin,end])
			else:
				#在类里面至少有一个点
				for classA in edge_class:
					for classB in edge_class:
						if begin in classA and end in classB and classA != classB:
							#属于不同的类：合并类
							edge_class.remove(classA)
							edge_class.remove(classB)
							classA.extend(classB)
							edge_class.append(classA)
							flag = 1
							break
						elif begin in classB and end in classB:
							#属于同一类，成环退出
							flag = 0
							break
						elif (begin in classB and end not in classB) or (begin not in classB and end in classB):
							#有一点在，另一点不在，每次均判断
							flag = 2
						else:
							flag = 4
					if flag == 1 or flag ==0:
						break
				if flag == 2 or flag == 4:
					#将另一个结点并入该类
					for classes in edge_class:
						if begin in classes:
							edge_class.remove(classes)
							classes.append(end)
							edge_class.append(classes)
							break
						elif end in classes:
							edge_class.remove(classes)
							classes.append(begin)
							edge_class.append(classes)
							break
						else:
							pass

		else:
			edge_class.append([begin,end])

		#第一退出条件：最小生成树生成完成（也就是屏幕中只有一个class）
		if len(edge_class) == 1 and len(set(edge_class[0])) == num_of_point:
			for allpoint in edge_class:
				for eachpoint in allpoint:
					templistx.append(location_list[eachpoint][0])
					templisty.append(location_list[eachpoint][1])
				#取最大的和最小的
				lefttop = [min(templistx),min(templisty)]
				rightbottom = [max(templistx),max(templisty)]
				boundingbox.append([lefttop,rightbottom])
			#print(boundingbox)
			#cv2.line(canvas,location_list[begin],location_list[end],(0,0,0),1)
			total_weight += distance[begin,end]
			#print(total_weight)
			#cvshow(canvas,4000)
			return boundingbox
			break

		#第二退出条件：达到阈值
		if distance[begin,end] >= threshold:
			for allpoint in edge_class:
				for eachpoint in allpoint:
					templistx.append(location_list[eachpoint][0])
					templisty.append(location_list[eachpoint][1])
				#取最大的和最小的
				lefttop = [min(templistx),min(templisty)]
				rightbottom = [max(templistx),max(templisty)]
				boundingbox.append([lefttop,rightbottom])
			#print(boundingbox)
			#print(total_weight)
			#cvshow(canvas,4000)
			return boundingbox
			break

		if flag != 0:
			list_path.append([begin,end])
			#cv2.line(canvas,location_list[begin],location_list[end],(0,0,0),1)
			total_weight += distance[begin,end]

		#cvshow(canvas)
	#cv2.destroyAllWindows()

#全局设置
eventlist = readCsv(["x","y","t","p"],"eventsUAV.txt",",",iftranspose=False)
#先读时间戳
timestamp = list(set(readCsv(["t"],"eventsUAV.txt",",",iftranspose=False,iflist=True)))
timestamp.sort()
threshold = 20
eventlist_copy = copy.deepcopy(eventlist)
video = cv2.VideoCapture("QQ20200509142514.mp4")
total_count = 0
for eachtime in range(len(timestamp)):
	ret,frame = video.read()
	eventlist = [x for x in eventlist_copy if x[2] == timestamp[eachtime]] #该部分用时过长
	a = datetime.datetime.now()
	selected = []
	for i in range(30):
		yes = random.choice(eventlist)
		selected.append(yes)
	selected = event2Point(selected)
	eachclass = selected.T
	distance = distanceGenerate(eachclass)
	boundingbox = Kruskal(distance,selected,threshold)
	for each in boundingbox:
		cv2.rectangle(frame, tuple(each[0]),tuple(each[1]), (0,0,255), 2)
	cv2.imshow("name",frame)
	if cv2.waitKey(1) & 0xFF == 27: #27位esc退出键
		break
	b = datetime.datetime.now()
	total_count += 1000*(b-a).total_seconds()

