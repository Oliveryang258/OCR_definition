import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required= True,
                help = "Path to the image t the scanned")
args = vars(ap.parse_args())

def cv_show(name,img):
	cv2.imshow(name,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def order_points(pts):
	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	# 获取输入坐标点
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 变换后对应坐标位置
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# 计算变换矩阵
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回变换后结果
	return warped

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation=inter)
	return resized

image = cv2.imread(args["image"])

#计算一个变换的比率后面变换完以后的轮廓可以放缩回来
ratio = image.shape[0] / 500.0
orig = image.copy()

#1，提取轮廓
image = resize(image,height=500)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#进行高斯滤波，剔除干扰项
gray = cv2.GaussianBlur(gray,(5,5),0)
#两个数值是双阈值检测的下阈值和上阈值，低于下阈值的被舍弃，高于上阈值的被认定为边界，中间连着边界才把保留
edged = cv2.Canny(gray,75,200)
cv_show('edged',edged)

#2，轮廓检测
cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
#下面这一句代码是将前面识别到的轮廓按照轮廓面积大小来排序，取前五个
cnts = sorted(cnts,key = cv2.contourArea,reverse= True)[:5]

for c in cnts:
	

