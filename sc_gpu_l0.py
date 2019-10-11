#-*- coding: utf-8 -*-

"""
This is an l0 scaling camouflage attack with GPU support
Author: Yufei Chen
Mail: yfchen@sei.xjtu.edu.cn
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import utils
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ScalingCamouflageGPU(object):
    """
    The ScalingCamouflage class.
    """
    def __init__(self, sourceImg=None, targetImg=None, **kwargs):
        """
        The constructor of the ScalingCamouflageGPU class.

        Args:
            sourceImg: The source image.
                       A numpy matrix shaped in (height1, width1, channels).
            targetImg: The target image.
                       A numpy matrix shaped in (height2, width2, channels).
            kwargs: Parameters of the scaling attack, include:
                    'func': the resizing function, e.g., cv2.resize;
                    'interpolation': the interpolation option,
                                     usually is an interger,
                                     e.g., cv2.INTER_NEAREST;
                    'L_dist': L-norm distance used as the metric;
                              (To be added in the future)
                    'eps': norm constraint of the perturbation, range:[0, 1];
                    'img_factor': map the pixel value [0, img_factor]
                                                 into [0, 1]
        """
        _, __, *channel = sourceImg.shape
        if not channel:
            self.sourceImg = sourceImg[:, :, np.newaxis]
        else:
            self.sourceImg = sourceImg

        _, __, *channel = targetImg.shape
        if not channel:
            self.targetImg = targetImg[:, :, np.newaxis]
        else:
            self.targetImg = targetImg

        # Initialize the parameters
        self.params = {'func': cv2.resize,
                       'interpolation': cv2.INTER_LINEAR,
                       'L_dist': 'L2',
                       'penalty': 1.,
                       'img_factor': 255.}
        keys = self.params.keys()
        # Set the parameters
        for key, value in kwargs.items():
            assert key in keys, ('Improper parameter %s, '
                                 'The parameter should in: '
                                 '%s' %(key, keys))
            self.params[key] = value

    def setResizeMethod(self, func=cv2.resize,
                        interpolation=cv2.INTER_NEAREST):
        """
        Set the resize method.

        Args:
            func: the resizing function, e.g., cv2.resize;
            interpolation: the interpolation option,
                           usually is an interger,
                           e.g., cv2.INTER_NEAREST;
        """
        self.params['func'] = func
        self.params['interpolation'] = interpolation

    def setSourceImg(self, sourceImg):
        """
        Set the source image.

        Args:
            sourceImg: The source image.
                       A numpy matrix shaped in (height, width, channels).
        """
        _, __, *channel = sourceImg.shape
        if not channel:
            self.sourceImg = sourceImg[:, :, np.newaxis]
        else:
            self.sourceImg = sourceImg

    def setTargetImg(self, targetImg):
        """
        Set the target image.

        Args:
            targetImg: The target image.
                       A numpy matrix shaped in (height, width, channels).
        """
        _, __, *channel = targetImg.shape
        if not channel:
            self.targetImg = targetImg[:, :, np.newaxis]
        else:
            self.targetImg = targetImg

    def estimateConvertMatrix(self, inSize, outSize):
        """
        Estimate the convert matrix.

        Args:
           inSize: The original input size before resizing.
           outSize: The output size after resizing.

        Returns:
            convertMatrix: The estimated convert matrix.
        """
        # Input a dummy test image (An identity matrix * 255).
        inputDummyImg = (self.params['img_factor'] *
                         np.eye(inSize)).astype('uint8')
        outputDummyImg = self._resize(inputDummyImg,
                                      outShape=(inSize, outSize))
        # Scale the elements of convertMatrix within [0,1]
        convertMatrix = (outputDummyImg[:,:,0] /
                (np.sum(outputDummyImg[:,:,0], axis=1)).reshape(outSize, 1))

        return convertMatrix

    def _resize(self, inputImg, outShape=(0,0)):
        """
        Resize the input image.

        Args:
            inputImg: The input image.
                      A numpy matrix shaped in (height1, width1, channels).
            outShape: The shape of the resized imaged,
                      formatted as (height, width).

        Returns:
            outputImg: The output image.
                       A numpy matrix shaped in (height2, width2, channels).
        """
        func = self.params['func']
        interpolation = self.params['interpolation']

        # PIL's resize method can only be performed on the PIL.Image object.
        if func is Image.Image.resize:
            inputImg = Image.fromarray(inputImg)
        if func is cv2.resize:
            outputImg = func(inputImg, outShape, interpolation=interpolation)
        else:
            outputImg = func(inputImg, outShape, interpolation)
            outputImg = np.array(outputImg)
        if len(outputImg.shape) == 2:
            outputImg = outputImg[:,:,np.newaxis]
        return np.array(outputImg)

    def _getPerturbationGPU(self, convertMatrixL, convertMatrixR, source, target):
        """
        Generate perturbation for scaling attack from source image to target image.

        Args:
            sourceVec: The source vector.
                       A numpy matrix shaped in (n, 1).
            targetVec: The target vector.
                       A numpy matrix shaped in (m, 1).

        Returns:
            perturb: The perturbation vector.
                     A numpy amtrix shaped in (n, 1).
        """
        penalty_factor = self.params['penalty']
        p, q, c = source.shape
        a, b, c = target.shape
        convertMatrixL = tf.constant(convertMatrixL, dtype=tf.float32)
        convertMatrixR = tf.constant(convertMatrixR, dtype=tf.float32)

        modifier_init = np.zeros(source.shape)

        source = tf.constant(source, dtype=tf.float32)
        target = tf.constant(target, dtype=tf.float32)

        modifier = tf.Variable(modifier_init, dtype=tf.float32)

        #w = (tf.tanh(modifier) + 1) * 0.5

        #attack = tf.compat.v1.placeholder(tf.float32, shape=source.shape)


        feasible_area = np.ones(source.shape)
        feasible_area_ph = tf.placeholder(dtype=tf.float32, shape=source.shape)

        prev_obj = np.inf
        while True:

            attack = source*(1-feasible_area_ph) + 0.5*(tf.tanh(modifier)+1)*feasible_area_ph

            x = tf.reshape(attack, [p, -1])
            x = tf.matmul(convertMatrixL, x)
            x = tf.reshape(x, [-1, q, c])
            x = tf.transpose(x, [1, 0, 2])
            x = tf.reshape(x, [q, -1])
            x = tf.matmul(convertMatrixR, x)
            x = tf.reshape(x, [-1, a, c])
            output = tf.transpose(x, [1, 0, 2])

            delta_1 = attack - source
            delta_2 = output - target

            obj1 = tf.reduce_sum(tf.square(delta_1))  / (p*q)
            obj2 = penalty_factor * tf.reduce_sum(tf.square(delta_2)) / (a*b)

            obj = obj1 + obj2

            outgrad = tf.gradients(obj, [modifier])[0]

            max_iteration = 2000
            with tf.compat.v1.Session() as sess:
                tf.assign(modifier, modifier_init)
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
                op = optimizer.minimize(obj, var_list=[modifier])
                prev = np.inf
                sess.run(tf.compat.v1.global_variables_initializer())
                print('------------')
                print('Area:')
                area_sum = np.sum(feasible_area)/c
                print(area_sum)
                for i in range(max_iteration):
                    _, obj_value, modifier_init1 = sess.run([op, obj, modifier], feed_dict={feasible_area_ph:feasible_area})
                    if i % 1000 == 0:
                        print(obj_value)
                        if obj_value > 0.999 * prev:
                            break
                        prev = obj_value
                obj1_value, obj2_value, obj_value = sess.run([obj1, obj2, obj], feed_dict={feasible_area_ph:feasible_area})
                print("Obj1:", obj1_value, ", Obj2:", obj2_value, ", Obj:", obj_value)
                if obj_value > prev_obj * 1.2:
                    return attack_opt_old

                prev_obj = obj_value
                counter = 0
                while True:
                    grads, delta1, obj_value = sess.run([outgrad, delta_1, obj2], feed_dict={feasible_area_ph:feasible_area})
                    if obj_value > prev_obj * 1.05:
                        print('Obj_value:', obj_value, ', Previous Obj_value', prev_obj)
                        break
                    else:
                        counter += 1
                        print('Reduce:', counter)
                        prev_obj = obj_value

                    change_measure = np.sum(np.abs(grads * delta1)*feasible_area, axis=2)
                    tmp = change_measure[np.where(change_measure>0)].flatten()
                    print(len(tmp))
                    if len(tmp) == 0:
                        attack_opt = attack.eval(feed_dict={feasible_area_ph:feasible_area})
                        return attack_opt
                    else:
                        threshold = np.percentile(tmp, 10)
                    valid = np.ones((p,q))
                    valid[np.where(change_measure<=threshold)]=0
                    feasible_area = np.repeat(valid[:, :, np.newaxis], c, axis=2)
                    attack_opt_old = attack.eval(feed_dict={feasible_area_ph:feasible_area})


    def attack(self):
        """
        Launch the scaling attack

        Returns:
            attackImg: The attack image crafted along a single direction.
        """

        sourceImg = self.sourceImg
        targetImg = self.targetImg

        sourceHeight, sourceWidth, sourceChannel = sourceImg.shape
        targetHeight, targetWidth, targetChannel = targetImg.shape

        convertMatrixL = self.estimateConvertMatrix(sourceHeight, targetHeight)
        convertMatrixR = self.estimateConvertMatrix(sourceWidth, targetWidth)
        img_factor = self.params['img_factor']
        sourceImg = sourceImg / img_factor
        targetImg = targetImg / img_factor

        # Add progress information

        source = sourceImg
        target = targetImg
        self.info()
        attackImg = self._getPerturbationGPU(convertMatrixL,
                                             convertMatrixR,
                                             source, target)

        #perturb = perturb_opt

        #attackImg = sourceImg + perturb
        print(np.max(attackImg))
        print(np.min(attackImg))
        print('Done! :)')
        return np.uint8(attackImg * img_factor)

    def info(self):
        """
        print out the attack info.
        """
        if self.params['func'] is cv2.resize:
            func_name = 'cv2.resize'
            inter_dict = ['cv2.INTER_NEAREST',
                          'cv2.INTER_LINEAR',
                          'cv2.INTER_CUBIC',
                          'cv2.INTER_AREA',
                          'cv2.INTER_LANCZOS4']
            inter_name = inter_dict[self.params['interpolation']]
        elif self.params['func'] is Image.Image.resize:
            func_name = 'PIL.Image.resize'
            inter_dict= ['PIL.Image.NEAREST',
                         'PIL.Image.LANCZOS',
                         'PIL.Image.BILINEAR',
                         'PIL.Image.BICUBIC']
            inter_name = inter_dict[self.params['interpolation']]

        # Note: The shape read from the image
        #       matrix is (height,width,channel)!
        sourceShape = (self.sourceImg.shape[1],
                       self.sourceImg.shape[0],
                       self.sourceImg.shape[2])

        targetShape = (self.targetImg.shape[1],
                       self.targetImg.shape[0],
                       self.targetImg.shape[2])

        print('------------------------------------')
        print('**********|Scaling Attack|**********')
        print('Source image size: %s' %str(sourceShape))
        print('Target image size: %s' %str(targetShape))
        print()
        print('Resize method: %s' %func_name)
        print('interpolation: %s' %inter_name)
        print('------------------------------------')


def test():
    sourceImgPath = sys.argv[1]
    targetImgPath = sys.argv[2]
    attackImgPath = sys.argv[3]

    sourceImg = utils.imgLoader(sourceImgPath)
    targetImg = utils.imgLoader(targetImgPath)

    print("Source image: %s" %sourceImgPath)
    print("Target image: %s" %targetImgPath)

    sc_gpu = ScalingCamouflageGPU(sourceImg,
                  targetImg,
                  func=cv2.resize,
                  interpolation=cv2.INTER_LINEAR,
                  penalty=1,
                  img_factor=255.)

    attackImg = sc_gpu.attack()
    utils.imgSaver(attackImgPath, attackImg)
    print("The attack image is saved as %s" %attackImgPath)

if __name__ == '__main__':
    test()
