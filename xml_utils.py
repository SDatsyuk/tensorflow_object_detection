import os
# import xml.etree.ElementTree as ET
from lxml import etree as ET

def create_xml_main(folder_text, filename_text, image_shape, database_text='Unknown'):
    r"""
    <annotation>
        <folder>imgs</folder>
        <filename>2.jpg</filename>
        <path>C:\Users\Super\Desktop\Nik\labelimg2\imgs\2.jpg</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>1536</width>
            <height>2048</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
            <name>pack</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>208</xmin>
                <ymin>695</ymin>
                <xmax>273</xmax>
                <ymax>777</ymax>
            </bndbox>
        </object>
    </annotation>

    """
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder')
    folder.text = folder_text

    filename = ET.SubElement(annotation, 'filename')
    filename.text = filename_text

    path = ET.SubElement(annotation, 'path')
    path.text = os.path.abspath(filename_text)

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = database_text

    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(image_shape[0])
    height = ET.SubElement(size, 'height')
    height.text = str(image_shape[1])
    depth = ET.SubElement(size, 'depth')
    depth.text = str(3)

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = str(0)

    return annotation

def create_annotation_object(pack_name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated='0', difficult='0'):    
    root = ET.Element('root')

    object_tag = ET.SubElement(root, 'object')
    name_tag = ET.SubElement(object_tag, 'name')
    name_tag.text = pack_name

    pose_tag = ET.SubElement(object_tag, 'pose')
    pose_tag.text = pose

    truncated_tag = ET.SubElement(object_tag, 'truncated')
    truncated_tag.text = truncated

    difficult_tag = ET.SubElement(object_tag, "difficult")
    difficult_tag.text = difficult

    bndbox = ET.SubElement(object_tag, 'bndbox')

    xmin_tag = ET.SubElement(bndbox, 'xmin')
    xmin_tag.text = str(xmin)
    ymin_tag = ET.SubElement(bndbox, 'ymin')
    ymin_tag.text = str(ymin)

    xmax_tag = ET.SubElement(bndbox, 'xmax')
    xmax_tag.text = str(xmax)
    ymax_tag = ET.SubElement(bndbox, 'ymax')
    ymax_tag.text = str(ymax)

    return root

def save_xml(data, path, filename):
    with open(os.path.join(path, filename), 'wb') as f:
        f.write(data)
    return True