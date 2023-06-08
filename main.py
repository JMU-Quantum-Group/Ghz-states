# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:37:05 2022

@author: 好梦难追
"""
# 2~(n-1) qubit ghz states as training data, detecting k-separable of n-qubit ghz states
import pandas as pd
from keras.utils import to_categorical
from generate_data import *
from model import *
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
class gr_data():
    def __init__(self, n, k, number1):
        self.n = n
        self.k = k
        self.number1 = number1
    def gen_data(self):
        gen = generate(self.k,4, self.number1)
        for i in range(7):
            gen.get_data()
            gen.__next__()
        #return data
    def read_data(self):
        for i in range(4, 11):
            train_features_0 = np.loadtxt('./data/' + repr(i) + '-qubit-' + repr(self.k) + '-0-' + repr(self.number1) +'-features.txt', skiprows=0)
            train_features_1 = np.loadtxt('./data/' + repr(i) + '-qubit-' + repr(self.k) + '-1-' + repr(self.number1) +'-features.txt', skiprows=0)
            la = np.zeros((2*len(train_features_0), 1))
            la[len(train_features_0):] = 1
            un_da = np.loadtxt('./data/' + repr(i) + '-qubit-' + repr(k) + '-un-' + repr(self.number1) + '-features.txt',skiprows=0)
            bound_states = np.loadtxt(
                './data/' + repr(i) + '-qubit-' + repr(self.k) + '-bound-features.txt',
                skiprows=0)
            if i == 4:
                train_data = np.concatenate((train_features_0, train_features_1), axis=0)
                train_label = la
                untrain_data = un_da
                bound_data = bound_states
            else:
                train_data = np.concatenate((train_data, train_features_0, train_features_1), axis=0)
                train_label = np.concatenate((train_label, la), axis=0)
                untrain_data = np.concatenate((untrain_data, un_da), axis=0)
                bound_data = np.concatenate((bound_data, bound_states), axis=0)
        #untrain = np.loadtxt('./data/' + repr(self.n) + '-qubit-' + repr(k) + '-un-' + repr(self.number1) + '-features.txt',skiprows=0)
        train_la = to_categorical(train_label)
        cb_la = np.zeros((len(bound_states), 1))
        cb_la[int(len(bound_states)/2):] = 1
        cb_la= to_categorical(cb_la)
        per = list(np.random.permutation(len(train_la)))
        train_la = train_la[per]
        train_data = train_data[per]
        return train_data, train_la, untrain_data
    def read_bound(self, bn):
        bound_states = np.loadtxt('./data/' + repr(bn) + '-qubit-' + repr(self.k) + '-bound-features.txt',skiprows=0)
        cb_la = np.zeros((len(bound_states), 1))
        cb_la[int(len(bound_states)/2):] = 1
        cb_la = to_categorical(cb_la)
        return bound_states, cb_la

class Run_Model():
    def __init__(self, k, number, Epoch, epoch,lamd):
        self.Epoch, self.epoch = Epoch, epoch
        self.k = k
        self.acc = pd.DataFrame(np.zeros((10, 4)),
                                columns=["ssl_acc", "slk_acc", "n_qubit", "number1"])
        self.number = number
        self.lamd = lamd
    def plot_roc(self, y_pred_keras, y_pred_keras2, y_test, n,fig):
        y = np.argmax(y_test, axis=1)
        lab = ["a", "b"," c"," d"," e", "f"]
        plt_index = n-3
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y, y_pred_keras.ravel(), drop_intermediate=False)
        auc_keras = auc(fpr_keras, tpr_keras)
        fpr_keras2, tpr_keras2, thresholds_keras2 = roc_curve(y, y_pred_keras2.ravel(), drop_intermediate=False)
        auc_keras2 = auc(fpr_keras2, tpr_keras2)
        ax = fig.add_subplot(2, 2, plt_index)
        ax.plot(fpr_keras, tpr_keras,  linestyle=':',
                label='SSL(Area = {:.3f})'.format(auc_keras))
        ax.plot(fpr_keras2, tpr_keras2,  linestyle='-.',
                label='SLK(Area = {:.3f})'.format(auc_keras2))
        ax.set_xlabel('False positive rate'+'\n('+lab[plt_index-1]+')')
        ax.set_ylabel('True positive rate')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_title(repr(n) +'-qubit, ' + repr(self.k)+'-separable, N=' + repr(self.number))
        ax.legend(loc="lower left")
        plt.tight_layout(pad=1, h_pad=3.0, w_pad=3.0)
        plt.show()
        plt.savefig('./result-fig/fig-roc-' + repr(self.lamd) + '-' + repr(self.number) + '.png')
    def run_model(self):
        i, k, acc = 0, self.k, self.acc
        Epoch, epoch = self.Epoch, self.epoch
        #fig = plt.figure(figsize=(12, 9))
        gr = gr_data(4, k, self.number)

        st = ST_Model()
        slk_model = st.stmodel()
        ssl_model = st.stmodel()
        la_da, la_la, un_da = gr.read_data()
        label_size = len(la_la)
        slk_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.003),
                         metrics=['accuracy'])
        slk_model.fit(la_da, la_la, epochs=epoch, validation_split=0, batch_size=label_size)

        ubp = slk_model.predict_proba(un_da)
        UB = slk_model.predict_classes(un_da)
        un_data = [un_da[k:k + 6 * 30] for k in range(0, len(un_da), 6 * 30)]
        bas_ubp = [ubp[k:k + 6 * 30] for k in range(0, len(ubp), 6 * 30)]
        bas_ub= [UB[k:k + 6 * 30] for k in range(0, len(UB), 6 * 30)]
        for (i, tb) in zip(range(4, 7), [0.95, 0.95, 0.9]):
            index = np.argwhere(bas_ubp[i] >= tb)[:, 0]
            uub = to_categorical(bas_ub[i], num_classes=2)
            if i == 4:
                ub = uub[index]
                un_d = un_data[i][index]
            else:
                ub = np.concatenate((ub, uub[index]), axis=0)
                un_d = np.concatenate((un_d, un_data[i][index]), axis=0)
        '''
        bas = [ubp[k:k + 3] for k in range(0, len(ubp), 3)]
        ub_average = 1 / 3 * (np.sum(bas, axis=1))
        ub = np.argmax(ub_average, axis=1)
        # ub_average=Sharpen(ub_average,T=0.5)
        index = np.argwhere(ub_average > 0.85)[:, 0]
        index = Index(index, 2)
        ub = to_categorical(ub, num_classes=None)
        ub = QB(ub, 2)
        ub = ub[index]
        '''
        ul = self.lamd
        def mycrossentropy(y_true, y_pred, L_S=label_size, lamda= ul):
            cc = tf.keras.losses.CategoricalCrossentropy()
            loss1 = cc(y_true[0:L_S], y_pred[0:L_S])
            loss2 = cc(y_true[L_S:], y_pred[L_S:])
            loss = loss1 + lamda * loss2
            return loss
        X = np.concatenate((la_da, un_d), axis=0)
        Y = np.concatenate((la_la, ub), axis=0)
        ssl_model.compile(loss=mycrossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
        ssl_model.fit(X, Y, epochs=Epoch, validation_split=0, batch_size=len(Y))
        #slk_model.save('slk_model_k'+repr(k)+'_m2_' + repr(self.number) + '.h5')
        #slk_model.save('slk_model_k'+repr(k)+'m2_' + repr(self.number) + '.h5')
        i = 0
        for bn in range(4, 11):
            cb_da, cb_la = gr.read_bound(bn=bn)
            sl_acc = slk_model.evaluate(cb_da, cb_la)[1]
            sll_acc = ssl_model.evaluate(cb_da, cb_la)[1]
            bpk_sl = slk_model.predict_classes(cb_da)
            bpk_sll = ssl_model.predict_classes(cb_da)
            #bk_sl = ge.calculate_boundary(bpk=bpk_sl)
            #bk_sll = ge.calculate_boundary(bpk=bpk_sll)
            acc.iloc[i, 0], acc.iloc[i, 1], acc.iloc[i, 2], acc.iloc[i, 3], = sll_acc, sl_acc,  bn, self.number
            i += 1
            #y_pred_keras1 = ssl_model.predict(cb_da)[:, 1]
            #y_pred_keras2 = slk_model.predict(cb_da)[:, 1]
            acc.to_csv('./result/acc-k'+repr(k)+'m2-'+repr(ul)+'-' + repr(self.number) + '.csv')
            #self.plot_roc(y_pred_keras1, y_pred_keras2,  cb_la, bn, fig)
        return acc
if __name__ == '__main__':
    lamda = 0.11
    for (number, Epoch, epoch) in zip([100], [70], [90]):
    #for (number, Epoch, epoch) in zip([200, 1000, 2000], [200, 250, 300, 450], [160, 180, 200, 300]):
        k = 4
        #gr = gr_data(11, k, number)
        #gr.gen_data() #generate data
        run = Run_Model(k, number, Epoch, epoch, lamda)
        acc = run.run_model()
