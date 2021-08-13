import numpy as np

def order_points(pts):
	# 初始化矩形4个顶点的坐标
	rect = np.zeros((4, 2), dtype='float32')
	# 坐标点求和 x+y
	s = pts.sum(axis = 1)
	# np.argmin(s) 返回最小值在s中的序号
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# diff就是后一个元素减去前一个元素  y-x
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# 返回矩形有序的4个坐标点
	return rect