import colorsys
import random

def get_colors():
	N = len(labels_to_names)
	HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
	RGB_tuples = list(map(lambda x: tuple(255*np.array(colorsys.hsv_to_rgb(*x))), HSV_tuples))
	random.shuffle(RGB_tuples)
	return RGB_tuples

def 