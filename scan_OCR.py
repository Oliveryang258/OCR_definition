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

#para: pts:一个包含四个点坐标，顺序位置的列表
#result: rect:一个顺序固定的列表，[左上，右上，右下，左下]
def order_points(pts):
	# 一共4个坐标点
	#创建一个4x2的矩阵来存储结果，格式为[[x1,y1], [x2,y2], ...]
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下(使用坐标和)(axis = 1)意味着沿着行操作,在这个矩阵中就是对一个点的x和y操作
	s = pts.sum(axis = 1)
	#左上角的点，其x和y都最小，所以x+y的和也最小
	rect[0] = pts[np.argmin(s)]
	# 右下角的点，其x和y都最大，所以x+y的和也最大
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下(使用坐标差) diff函数默认是用后一列来减去前一列，所以是y-x
	diff = np.diff(pts, axis = 1)
	# 右上角的点，其y最小，x最大，所以y-x的差最小（是个很大的负数）
	rect[1] = pts[np.argmin(diff)]
	# 左下角的点，其y最大，x最小，所以y-x的差最大（是个很大的正数）
	rect[3] = pts[np.argmax(diff)]

	return rect

#para: img：原始输入的图像
#      pts：得到的外轮廓点集列表
def four_point_transform(image, pts):
	# 获取输入坐标点
	# 1,先排序获得一个顺序固定的列表
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算输入的w和h值
	# 2,取两者中最大的长度和宽度作为最终的长宽，防止信息丢失
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 3,变换后对应坐标位置
	dst = np.array([
		[0, 0],#新的左上角默认为[0,0]
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# 4,计算变换矩阵
	# 这是最核心的一步。OpenCV会比较“原始四边形的角点(rect)”和“目标矩形的角点(dst)”
    # 然后计算出一个 3x3 的变换矩阵 M。这个矩阵包含了所有拉伸、旋转、倾斜的数学信息。
	M = cv2.getPerspectiveTransform(rect, dst)
	#5,应用变换矩阵
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
#因为opencv4.0版本的原因，findContours函数只返回两个量，轮廓是第一项
cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
#下面这一句代码是将前面识别到的轮廓按照轮廓面积大小来排序，只保留前五个最大的
cnts = sorted(cnts,key = cv2.contourArea,reverse= True)[:5]

#3，遍历轮廓，获取外层轮廓
for c in cnts:
	#计算轮廓近似，因为前面得到的轮廓不一定是一个完整的封闭图案，可能是很多不连续的点，所以这里把点集近似成一个封闭图形
	peri = cv2.arcLength(c,True)
	#c表示前面输入的点集
	#epsilon表示从原始轮廓到近似轮廓的最大距离，是一个精确度参数,越小越精致，一般使用长度的百分之几作为这个值
	#True表示是封闭图像
	approx = cv2.approxPolyDP(c,0.02*peri,True)

	#因为这里的目的是提取外面轮廓，所以当这个近似图像得到四个点的时候，即为矩形，就可以拿出来特判了
	#len函数的作用是判断近似轮廓中包含多少个顶点
	if len(approx) == 4:
		screenCnt = approx
		break

#画出最外圈的框
if screenCnt is not None:
	cv2.drawContours(image,[screenCnt],-1,(0,0,255),2)
	cv_show('outline',image)
else:
	print("无法找到矩形轮廓")

#4,透视变换，将外框矩形摆正，这里这个reshape(4,2)是点，每个点有xy两个参数，乘以ratio是为了把这几个点还原到原始的输入图像当中
warped = four_point_transform(orig,screenCnt.reshape(4,2)*ratio)

#5,对变换后的图像再进行一个二值处理
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped,100,255,cv2.THRESH_BINARY)[1]
cv2.imwrite('scan.jpg',ref)

#6,展示结果
cv_show('original',resize(orig,height= 600))
cv_show('scanned',resize(ref,height= 600))
#后续对这个检测到的图像使用ocr检测

