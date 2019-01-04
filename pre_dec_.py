from __future__ import print_function

import glob
import os
import shutil
import sys
import PIL.Image as Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

sys.path.append(os.getcwd()+'/text-detection-ctpn-master')
#print('..'+os.getcwd())
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import _get_blobs
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.rpn_msr.proposal_layer_tf import proposal_layer


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale,score):
    base_name = image_name.split('/')[-1]
    im0 = Image.open(image_name)
    j=0
    for box in boxes:

        if score[j] >= 0.9:
            color = (0, 255, 0)
        elif score[j] >= 0.8:
            color = (255, 0, 0)
        elif score[j] >= 0.7:
            color = (0, 0, 255)
        else:
            color = (255, 255, 255)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[0]), int(box[3])), color, 2)
        cv2.line(img, (int(box[0]), int(box[3])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[2]), int(box[3])), (int(box[2]), int(box[1])), color, 2)
        cv2.line(img, (int(box[2]), int(box[1])), (int(box[0]), int(box[1])), color, 2)
        '''
        min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        img_box = (min_x,min_y,max_x,max_y)
        mini_img = im0.crop(img_box)
        mini_img.save('../train/data/' + base_name.split('.')[0] + '_' + str(j) + '.png')

       

        line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
        f.write(line)

        '''
        j = j + 1
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("upload/", base_name), img)
def draw_line(img, image_name, boxes, scale):
    base_name = image_name.split('/')[-1]
    #im0 = Image.open(image_name)
    im0=cv2.imread(image_name,cv2.IMREAD_GRAYSCALE)
    predition_result={}
    predition_result['box']=[]
    predition_result['result'] = []
    predition_result['file'] = []
    j=0

    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue
        if box[8] >= 0.9:
            color = (0, 255, 0)
        elif box[8] >= 0.8:
            color = (255, 0, 0)
        elif box[8] >= 0.7:
            color = (0, 0, 255)
        else:
            color=(255,255,255)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)
        j=j+1
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("upload/crop_img/", base_name), img)
    #return predition_result

if __name__ == '__main__':
    cfg_from_file('text-detection-ctpn-master/ctpn/text.yml')
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    with gfile.FastGFile('text-detection-ctpn-master/data/ctpn.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')

    im_names = glob.glob(os.path.join('upload/origin', '*.png')) + \
               glob.glob(os.path.join('upload/origin', '*.jpg'))

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        img = cv2.imread(im_name)
        img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        blobs, im_scales = _get_blobs(img, None)
        if cfg.TEST.HAS_RPN:
            im_blob = blobs['data']
            blobs['im_info'] = np.array(
                [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
                dtype=np.float32)
        cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)
        scores = rois[:, 0]
        boxes = rois[:, 1:5] / im_scales[0]

        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])

        #draw_boxes(img, im_name, boxes, scale,scores)
        draw_line(img, im_name, boxes, scale)
