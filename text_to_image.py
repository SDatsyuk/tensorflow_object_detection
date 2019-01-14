import io
import os
import cv2
import sys
import json
import base64
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from object_detection.utils import label_map_util
from object_detection.utils.visualization_utils import STANDARD_COLORS


def parse_recognition_result(res, labels):
    """
    [[697, 1460, 786, 1549, 0, 1], [733, 1272, 836, 1396, 0, 4], [791, 1459, 876, 1551, 0, 1], 
    [881, 1460, 962, 1551, 0, 1], [931, 1281, 1026, 1401, 0, 6], [976, 1459, 1057, 1549, 0, 1], 
    [1036, 1275, 1133, 1399, 0, 6], [1057, 1460, 1144, 1553, 0, 3], [1139, 1454, 1226, 1551, 0, 2], 
    [1171, 1269, 1273, 1385, 0, 6], [1241, 1453, 1330, 1550, 0, 2], [1271, 1275, 1365, 1391, 0, 6], 
    [1332, 1456, 1414, 1549, 0, 2], [1377, 1275, 1467, 1391, 0, 7], [1415, 1457, 1504, 1554, 0, 3], 
    [1467, 1271, 1555, 1391, 0, 7], [1517, 1467, 1606, 1556, 0, 4], [1619, 1272, 1712, 1393, 0, 7], 
    [1691, 1468, 1786, 1564, 0, 4], [1728, 1280, 1824, 1391, 0, 7], [1790, 1473, 1884, 1567, 0, 5], 
    [1813, 1282, 1920, 1394, 0, 8], [1880, 1473, 1974, 1568, 0, 5], [1909, 1279, 2026, 1392, 0, 9], 
    [1971, 1475, 2064, 1569, 0, 5], [2020, 1278, 2130, 1393, 0, 10]]
    """
    ret = {}
    box_to_color_map = {}
    for i in res:
        idx = labels[i[-1]]['name']
        box_to_color_map[idx] = STANDARD_COLORS[
              i[-1] % len(STANDARD_COLORS)]
        if idx in ret:
            ret[idx] += 1
        else:
            ret[idx] = 1

    return ret, box_to_color_map

def load_labels(file, num_classes=12):
    label_map = label_map_util.load_labelmap(file)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print(category_index)
    return category_index


def text2png(text, colors, fullpath=None, color = "#000", bgcolor = "#222", fontfullpath = None, fontsize = 16, leftpadding = 3, rightpadding = 3, width = 240):
    REPLACEMENT_CHARACTER = u'\uFFFD'
    NEWLINE_REPLACEMENT_STRING = ' ' + REPLACEMENT_CHARACTER + ' '

    #prepare linkback
    linkback = "created via http://ourdomain.com"
    fontlinkback = ImageFont.truetype(fontfullpath, 8)
    linkbackx = fontlinkback.getsize(linkback)[0]
    linkback_height = fontlinkback.getsize(linkback)[1]
    #end of linkback

    font = ImageFont.load_default() if fontfullpath == None else ImageFont.truetype(fontfullpath, fontsize)
    # text = text.replace('\n', NEWLINE_REPLACEMENT_STRING)

    lines = []
    line = u""
    print(text)
    width = max([font.getsize(i)[0] for i in text.split('\n')]) + leftpadding + rightpadding
    print(width)
    for word in text.split('\n'):
        print(word)

        if word == REPLACEMENT_CHARACTER: #give a blank line
            lines.append( line[1:] ) #slice the white space in the begining of the line
            line = u""
            lines.append( u"" ) #the blank line
        elif font.getsize( line + ' ' + word )[0] <= (width - rightpadding - leftpadding):
            line += ' ' + word
        else: #start a new line
            lines.append( line[1:] ) #slice the white space in the begining of the line
            line = u""

            #TODO: handle too long words at this point
            line += ' ' + word #for now, assume no word alone can exceed the line width

    if len(line) != 0:
        lines.append( line[1:] ) #add the last line

    line_height = font.getsize(text)[1]
    img_height = line_height * (len(lines) + 1)

    img = Image.new("RGBA", (width, img_height), bgcolor)
    draw = ImageDraw.Draw(img)

    y = 0
    print(colors)
    print(lines)
    print(line[0].split(":")[0])
    for i, line in enumerate(lines):
        if line:
            draw.text( (leftpadding, y), line, colors[line.split(":")[0]], font=font)
        y += line_height



    # add linkback at the bottom
    # draw.text( (width - linkbackx, img_height - linkback_height), linkback, color, font=fontlinkback)

    return img


if __name__ == "__main__":

    labels = load_labels('maps/philmor48.pbtxt', 48)

    d = [[697, 1460, 786, 1549, 0, 1], [733, 1272, 836, 1396, 0, 4], [791, 1459, 876, 1551, 0, 1], 
    [881, 1460, 962, 1551, 0, 1], [931, 1281, 1026, 1401, 0, 6], [976, 1459, 1057, 1549, 0, 1], 
    [1036, 1275, 1133, 1399, 0, 6], [1057, 1460, 1144, 1553, 0, 3], [1139, 1454, 1226, 1551, 0, 2], 
    [1171, 1269, 1273, 1385, 0, 6], [1241, 1453, 1330, 1550, 0, 2], [1271, 1275, 1365, 1391, 0, 6], 
    [1332, 1456, 1414, 1549, 0, 2], [1377, 1275, 1467, 1391, 0, 7], [1415, 1457, 1504, 1554, 0, 3], 
    [1467, 1271, 1555, 1391, 0, 7], [1517, 1467, 1606, 1556, 0, 4], [1619, 1272, 1712, 1393, 0, 7], 
    [1691, 1468, 1786, 1564, 0, 4], [1728, 1280, 1824, 1391, 0, 7], [1790, 1473, 1884, 1567, 0, 5], 
    [1813, 1282, 1920, 1394, 0, 8], [1880, 1473, 1974, 1568, 0, 5], [1909, 1279, 2026, 1392, 0, 9], 
    [1971, 1475, 2064, 1569, 0, 5], [2020, 1278, 2130, 1393, 0, 10]]


    res, colors = parse_recognition_result(d, labels)
    print(res)

    s = ''

    for i in res:
        s += '{}: {}\n'.format(i, res[i])

    print(s)

    img = text2png(s, colors, fontfullpath="L_10646.TTF")
    cv2.imshow('s', np.array(img))
    cv2.waitKey(0)