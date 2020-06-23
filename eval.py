import os
# not printing tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# preferably use the non-display gpu
os.environ['CUDA_VISIBLE_DEVICES']='1'

import math
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
from model import LiteFlowNet
import argparse

from draw_flow import *


def pad_image(image):
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 1

    nh = int(math.ceil(h / 32.) * 32)
    nw = int(math.ceil(w / 32.) * 32)

    pad_i = np.zeros([nh, nw, c])
    pad_i[:h, :w, :c] = image.reshape(h, w, c)
    return pad_i


def main():
    tf.disable_eager_execution()

    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', required=True, action='store', nargs=1, dest='img1')
    parser.add_argument('--img2', required=True, action='store', nargs=1, dest='img2')
    parser.add_argument('--model', required=True, nargs=1, dest='model')
    parser.add_argument('--flow_dir', required=True, nargs=1, dest='flow_dir')
    parser.add_argument('-o', required=True, action='store', nargs=1, dest='out_dir')
    # parser.add_argument('--display_flow', default=True)

    args = parser.parse_args()
    img1 = args.img1[0]
    img2 = args.img2[0]
    flow_dir = args.flow_dir[0]
    model_path = args.model[0]
    out_dir = args.out_dir[0]

    sess = tf.Session()
    model = LiteFlowNet()
    tens1 = tf.placeholder(tf.float32, shape=[1, 1024, 1024, 1])
    tens2 = tf.placeholder(tf.float32, shape=[1, 1024, 1024, 1])
    out = model(tens1, tens2)

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    inp1 = Image.open(img1)
    inp2 = Image.open(img2)

    w, h = inp1.size[:2]
    inp1 = np.float32(np.expand_dims(pad_image(np.asarray(inp1)[..., ::-1]), 0)) / 255.0
    inp2 = np.float32(np.expand_dims(pad_image(np.asarray(inp2)[..., ::-1]), 0)) / 255.0

    # input in bgr format
    flow = sess.run(out, feed_dict={tens1: inp1, tens2: inp2})[0, :h, :w, :]

    # visualize optical flow
    flow_color = flow_to_color(flow, convert_to_bgr=False)
    flow_image = Image.fromarray(flow_color)
    # flow_image.show()
    out_path = os.path.join(out_dir, 'LFT_result1.png')
    flow_image.save(out_path)
    print(f'Resulting image has been saved to {out_path}')

    # save optical flow to file
    # flow_path = os.path.join(flow_dir, 'LFT_flow.npy')
    # flow_str = np.array(np.transpose(flow).transpose([1, 2, 0]), np.float32)
    # np.save(flow_np, flow_path)
    # objectOutput = open(args.flow, 'wb')
    # np.array([80, 73, 69, 72], np.uint8).tofile(objectOutput)
    # np.array([flow.shape[1], flow.shape[0]], np.int32).tofile(objectOutput)
    # np.array(np.transpose(flow).transpose([1, 2, 0]), np.float32).tofile(objectOutput)
    # objectOutput.close()
    # print(f'Flow has been saved to {flow_path}')


if __name__ == "__main__":
    main()