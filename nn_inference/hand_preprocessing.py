import cv2
import numpy as np
import argparse
import time

morph_size = 0
max_operator = 4
max_elem = 2
max_kernel_size = 21
max_gaussian_blur_kernel = 10
title_trackbar_operator_type = 'Operator:\n 0: Opening - 1: Closing  \n 2: Gradient - 3: Top Hat \n 4: Black Hat'
title_trackbar_element_type = 'Element:\n 0: Rect - 1: Cross - 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n + 1'
title_trackbar_gaussian_blue = 'Gaussian blur kernel size:\n x*x'
title_window = 'Morphology Transformations Demo'
morph_op_dic = {0: cv2.MORPH_OPEN, 1: cv2.MORPH_CLOSE, 2: cv2.MORPH_GRADIENT, 3: cv2.MORPH_TOPHAT, 4: cv2.MORPH_BLACKHAT}
morph_elem_op_dict = {0: cv2.MORPH_RECT, 1: cv2.MORPH_CROSS, 2: cv2.MORPH_ELLIPSE}

#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=True) # best

def trackbar_callback(val):
	global element
	morph_operator = cv2.getTrackbarPos(title_trackbar_operator_type, title_window)
	morph_size = cv2.getTrackbarPos(title_trackbar_kernel_size, title_window)
	morph_elem = 0
	val_type = cv2.getTrackbarPos(title_trackbar_element_type, title_window)
	morph_elem = morph_elem_op_dict[val_type]
	# element = cv2.getStructuringElement(morph_elem, (morph_size, morph_size))
	element = cv2.getStructuringElement(morph_elem, (2*morph_size + 1, 2*morph_size+1), (morph_size, morph_size))
	# operation = morph_op_dic[morph_operator]

def callback(val):
	pass


cv2.namedWindow(title_window)
# cv2.createTrackbar(title_trackbar_operator_type, title_window, 0, max_operator, trackbar_callback)
cv2.createTrackbar(title_trackbar_element_type, title_window, 0, max_elem, trackbar_callback)
cv2.createTrackbar(title_trackbar_kernel_size, title_window, 0, max_kernel_size, trackbar_callback)
cv2.createTrackbar(title_trackbar_gaussian_blue, title_window, 1, max_gaussian_blur_kernel, callback)


def mad(data, axis=None):
	return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def med(data, axis=None):
	return np.mean(data, axis)

def shadow_thresholding(img, back):
	D = np.absolute(np.subtract(img, back))
	MED = med(D)
	MAD = mad(D)
	MED_Fv = np.mean(img)
	MED_Bv = np.mean(back)
	A = MAD / MED_Fv
	B = (MED + MAD) / MED_Bv
	return A, B


def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False

def concat_contours(contours):
	LENGTH = len(contours)
	status = np.zeros((LENGTH,1))

	for i,cnt1 in enumerate(contours):
	    x = i    
	    if i != LENGTH-1:
	        for j,cnt2 in enumerate(contours[i+1:]):
	            x = x+1
	            dist = find_if_close(cnt1,cnt2)
	            if dist == True:
	                val = min(status[i],status[x])
	                status[x] = status[i] = val
	            else:
	                if status[x]==status[i]:
	                    status[x] = i+1

	unified = []
	print(status)
	if not status.any():
		return contours
	maximum = int(status.max())+1
	for i in range(maximum):
	    pos = np.where(status==i)[0]
	    if pos.size != 0:
	        cont = np.vstack(contours[i] for i in pos)
	        hull = cv2.convexHull(cont)
	        unified.append(hull)
	return unified




def preproc(frame, background):
	# frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame_hsv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
	cv2.imshow('hsv', frame_hsv)
	# orig = frame_hsv.copy()
	# frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
	frame = cv2.GaussianBlur(frame, (5,5), 0)

	fgmask = fgbg.apply(frame)
	ret, mask = cv2.threshold(fgmask, 5, 255, cv2.THRESH_BINARY)

	A, B = shadow_thresholding(frame, background)
	coef = A / B
	#print(1 / background[:,:,2])

	shadow_mask = 1 / background[:, :, 2] < coef
	
	mask[mask != 255] = 0
	# cv2.imshow('with shadow', mask)
	# mask[shadow_mask] = 0
	# cv2.imshow('without shadow', mask)
	fgmask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
	gaussian_kernel = cv2.getTrackbarPos(title_trackbar_gaussian_blue, title_window)
	gaussian_kernel = gaussian_kernel + 1 if gaussian_kernel % 2 == 0 else gaussian_kernel
	mask = cv2.blur(mask, (10, 10), (-1, -1))
	mask = cv2.GaussianBlur(mask,(gaussian_kernel, gaussian_kernel),0)

	dst = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)
	dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)
	cv2.imshow('ss', dst)
	# dst = cv2.dilate()

	# dst = cv2.Canny(dst, 100, 200)
	_, contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	# contours = concat_contours(contours)
	contours = [c for c in contours if cv2.contourArea(c) > 100]
	cv2.drawContours(frame, contours, -1, (0,255,0), 3)

	msk = dst > 0
	
	frame[msk == False] = 0
	return frame
	
def back_sub(cap, fgbg):
	_, background = cap.read()
	background = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
	time.sleep(0.5)
	while True:
		_, frame = cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
		# frame_hsv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)

		res = preproc(frame, background)
		
		cv2.imshow('origin', res)
		#cv2.imshow('back', mask)
		#cv2.imshow('thresh', dst)

		key = cv2.waitKey(30)

		if key & 0xFF == 27:
			break
		background = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
	cap.release()
	cv2.destroyAllWindows()

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--input', default=0, type=int, help='camera index')
	ap.add_argument("-f", '--file', default=None, help='path to input video files')

	args = vars(ap.parse_args())






	cap = cv2.VideoCapture(args['file']) if args['file'] else cv2.VideoCapture(args['input'])
	if not cap.isOpened():
		assert AttributeError, 'Cannot open camera or video file'

	

	element = None

	trackbar_callback(0)
	back_sub(cap, fgbg)





if __name__ == "__main__":
	main()