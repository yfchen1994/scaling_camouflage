#-*- coding: utf-8 -*-

"""
This is an l-inf scaling camouflage attack with GPU support
Author: Yufei Chen
Mail: yfchen@sei.xjtu.edu.cn
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import cvxpy
import dccp
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class ScalingCamouflageGPU(object):
    """
    The SCalingCamouflage class.
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
                       'penalty': 1,
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
            sourceVec: The source image.
                       A numpy matrix shaped in (n, 1).
            targetVec: The target image.
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

        old_modifier = np.arctanh((2*source-1)*0.9999999)

        source = tf.constant(source, dtype=tf.float32)
        target = tf.constant(target, dtype=tf.float32)

        modifier = tf.Variable(np.zeros(source.shape), dtype=tf.float32)

        assign_modifier = tf.placeholder(np.float32, source.shape)
        set_modifier = tf.assign(modifier, assign_modifier)

        attack = (tf.tanh(modifier) + 1) * 0.5

        x = tf.reshape(attack, [p, -1])
        x = tf.matmul(convertMatrixL, x)
        x = tf.reshape(x, [-1, q, c])
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [q, -1])
        x = tf.matmul(convertMatrixR, x)
        x = tf.reshape(x, [-1, a, c])
        output = tf.transpose(x, [1, 0, 2])

        tau1 = tf.placeholder(tf.float32, [])
        tau2 = tf.placeholder(tf.float32, [])

        delta1 = attack - source
        delta2 = output - target

        #obj1 = tf.reduce_sum(tf.square(perturb) + tf.abs(perturb - tau)) / (p*q)
        obj1 = tf.reduce_sum(tf.maximum(0.0, (tf.abs(delta1) - tau1))) / (p*q)
        #obj2 = penalty_factor*tf.reduce_sum(tf.square(delta2) + tf.maximum(0.0, (tf.abs(delta2) - tau2))) / (a*b)
        obj2 = penalty_factor*tf.reduce_sum(tf.square(delta2)) / (a*b)
        obj = obj1 + obj2

        max_iteration = 2000
        t1 = 1.0
        #t2 = 1.0
        actualtau1 = np.inf
        #actualtau2 = 1.0

        with tf.Session() as sess:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            op = optimizer.minimize(obj, var_list=[modifier])
            max_steps = 20
            sess.run(tf.global_variables_initializer())

            while t1 > 1./self.params['img_factor']:
        #    for step in range(max_steps):
        #        print(step)
        #        if step % 2 == 1:
        #        else:
        #            if t2 < 1./self.params['img_factor']:
        #                continue
        #            if actualtau2 < t2:
        #                t2 = actualtau2
        #                t2 *= 0.9

                sess.run(set_modifier, feed_dict={assign_modifier:old_modifier})
                prev = np.inf

                for i in range(max_iteration):
                    _, obj_value, delta1_value, delta2_value, old_modifier, obj1_eval, obj2_eval = sess.run([op, obj, delta1, delta2, modifier, obj1, obj2], feed_dict={tau1:t1})
                    if i % 1000 == 0:
                        if obj_value > prev*.9999:
                             break
                        prev = obj_value
                        print('Obj:', obj_value, ', Obj1:', obj1_eval, ', Obj2:', obj2_eval)

                actualtau1 = np.max(np.abs(delta1_value))
                if actualtau1 < t1:
                    t1 = actualtau1
                    t1 *= 0.9
                else:
                    break
        #        else:
        #            break
        #        actualtau2 = np.max(np.abs(delta2_value))
        #        print("Tau1:", t1, ", Tau2:", t2)
                print("Tau1:", t1)

            attack_opt = attack.eval()
            perturb_opt = delta1.eval()
            print('tau:', actualtau1)
            print('obj1:',obj1.eval(feed_dict={tau1:t1}),', obj2:',obj2.eval())
        return perturb_opt


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

        perturb = np.zeros((sourceHeight, sourceWidth, sourceChannel))

        # Add progress information

        source = sourceImg
        target = targetImg
        self.info()
        perturb_opt = self._getPerturbationGPU(convertMatrixL,
                                               convertMatrixR,
                                               source, target)

        perturb = perturb_opt

        attackImg = sourceImg + perturb
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
