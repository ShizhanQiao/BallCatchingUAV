import cv2
import numpy as np 
import sys
def cvshow(img):
	cv2.imshow("test",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def LCK(ix,iy,it,ksize,pic):
	#计算每个点的光流，拒绝矛盾方程组因为不可逆
	#预构建光流图
	optical_pic = np.zeros((pic.shape[0],pic.shape[1],3))
	#开始滑动窗口，步长=ksize
	for y in range(0,pic.shape[0],ksize):
		for x in range(0,pic.shape[1]):
			#竖着取点计算
			ix_part = ix[y:y+ksize,x:x+1]
			iy_part = iy[y:y+ksize,x:x+1]
			it_part = it[y:y+ksize,x:x+1]
			#构建矛盾方程组
			ix_part = ix_part.flatten()
			iy_part = iy_part.flatten()
			it_part = it_part.flatten()
			#判断分母是否为0，如果为0的话那么光流就是0
			delta = ix_part[0]*iy_part[1]-ix_part[1]*iy_part[0]
			if delta != 0:
				u = (-it_part[1]*iy_part[0]+it_part[0]*iy_part[1])/(ix_part[0]*iy_part[1]-ix_part[1]*iy_part[0])
				v = (-it_part[0]*ix_part[1]+it_part[1]*ix_part[0])/(ix_part[0]*iy_part[1]-ix_part[1]*iy_part[0])
			else:
				u = 0
				v = 0
			#填数，前两个代表方向，第三个代表大小，+为255，-为0
			strong = np.sqrt(u**2+v**2)
			xvalue = 255 if u > 0 else 0
			yvalue = 255 if v > 0 else 0
			optical_pic[y:y+ksize,x,0] = xvalue
			optical_pic[y:y+ksize,x,1] = yvalue
			optical_pic[y:y+ksize,x,2] = 255*strong/5
	optical_pic = np.array(optical_pic,dtype=np.uint8)
	return optical_pic

video = cv2.VideoCapture("person01_02_ground_throwing.avi")
#读进来一帧
ret,lastframe = video.read()
#转化为灰度图
lastframe = cv2.cvtColor(lastframe,cv2.COLOR_BGR2GRAY)
#求dI/dx
dIdx = cv2.Sobel(lastframe,cv2.CV_64F,1,0,ksize=3)
cv2.convertScaleAbs(dIdx)

#求dI/dy
dIdy = cv2.Sobel(lastframe,cv2.CV_64F,0,1,ksize=3)
cv2.convertScaleAbs(dIdy)

while True:
	ret, thisframe = video.read()
	thisframe = cv2.cvtColor(thisframe,cv2.COLOR_BGR2GRAY)
	thisframe_copy = thisframe.copy()
	thisframe = np.array(thisframe,dtype=np.float32)
	lastframe = np.array(lastframe,dtype=np.float32)
	#求dI/dt
	dIdt = thisframe - lastframe
	optical_pic = LCK(dIdx,dIdy,dIdt,2,thisframe_copy)
	#cv2.imshow("name",optical_pic)
	cvshow(optical_pic)
	if cv2.waitKey(1) & 0xFF == 27: #27位esc退出键
		break
	lastframe = thisframe
	#求dI/dx
	dIdx = cv2.Sobel(lastframe,cv2.CV_64F,1,0,ksize=3)
	cv2.convertScaleAbs(dIdx)
	#求dI/dy
	dIdy = cv2.Sobel(lastframe,cv2.CV_64F,0,1,ksize=3)
	cv2.convertScaleAbs(dIdy)