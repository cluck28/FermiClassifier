#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:42:43 2017

Program to do some data analysis and see how our net has been trained

@author: Chris
"""

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import random

from NeuralNet_nohiddenlayers import load_data

def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = pickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    datasets = load_data()
    test_set_x, test_set_y = datasets[2] #Pick index to choose valid or test data
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x)
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values[:10])
    print("Actual values for the first 10 examples in the test set:")
    print(test_set_y[:10].eval())
    return test_set_x, test_set_y.eval(), predicted_values

def Gauss(x,*p):
    return p[0]*numpy.exp(-(x-p[1])**2./(2.*p[2]**2.))


if __name__ == '__main__':
    #Get values and predictions
    x_vals, true_y_vals, pred_y_vals = predict() #Get values
    wrong = numpy.array([])
    mis_true_y = numpy.array([])
    mis_pred_y = numpy.array([])
    counter = 0
    for i in range(len(true_y_vals)):
        if true_y_vals[i] != pred_y_vals[i]:
            counter += 1
            wrong = numpy.append(wrong,i)
            mis_true_y = numpy.append(mis_true_y,true_y_vals[i])
            mis_pred_y = numpy.append(mis_pred_y,pred_y_vals[i])
    print counter/10000.
    print numpy.unique(mis_true_y)
    print numpy.unique(mis_pred_y)
    if counter == 0:
        index = 1
    else:
        index = int(random.choice(wrong))
    print(index)
    params = [1,0,pred_y_vals[index]/10.]     
    