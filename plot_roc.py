from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd




y_test = np.load('./Data/2-qubit/y_test' + repr(number3) + '.npy')
X_test = np.load('./Data/2-qubit/X_input' + repr(number3) + '.npy')
y = np.argmax(y_test, axis=1)
# np.load('./Data/2-qubit/test_SA_M3'+repr(number3)+'.npy')
# XX_input=np.load('./Data/2-qubit/XX_input'+repr(number3)+'.npy')
# %%绘制子图
fig = plt.figure(figsize=(12, 9))
ylims = [0.6, 0.6, 0.6, 0.6]
xlims_right = [0.3, 0.3, 0.3, 0.3]
colors = ['#9467bd', '#ff7f0e', '#2ca02c']
for (plt_index, K, jj, number1, number2) in zip(range(1, 5), KK[5:9], JJ[5:9], L[5:9], U[5:9]):
    alpha = x[jj]
    label_size = (K + 1) * number1
    def crossentropy(y_true, y_pred, L_S=label_size, lamda=alpha):  # batch_size标签数据
        cc = tf.keras.losses.CategoricalCrossentropy()
        loss1 = cc(y_true[0:L_S], y_pred[0:L_S])
        loss2 = cc(y_true[L_S:], y_pred[L_S:])
        loss = loss1 + lamda * loss2
        return loss
    su_model0 = keras.models.load_model('./max_model/mmodel_' + repr(number1) + '_' + repr(number2) + '.h5')
    su_model = keras.models.load_model(
        './max_model/pmodel_KK' + repr(K) + '+_' + repr(number1) + '_' + repr(number2) + '_' + repr(threshould) + '.h5')
    semi_model = keras.models.load_model(
        './max_model/mmodel_inter' + repr(jj) + 'KK' + repr(K) + '+_' + repr(number1) + '_' + repr(
            number2) + '_' + repr(threshould) + '.h5', custom_objects={'crossentropy': crossentropy})
    su_model.evaluate(X_test, y_test, batch_size=600)[1]
    # semi_model.evaluate(val_E,truth_E,batch_size =len(val_E))[1]
    y_pred_keras = semi_model.predict(X_test)[:, 1]
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y, y_pred_keras.ravel())
    auc_keras = auc(fpr_keras, tpr_keras)
    y_pred_keras2 = su_model.predict(X_test)[:, 1]
    fpr_keras2, tpr_keras2, thresholds_keras2 = roc_curve(y, y_pred_keras2.ravel())
    auc_keras2 = auc(fpr_keras2, tpr_keras2)
    y_pred_keras3 = su_model0.predict(X_test)[:, 1]
    fpr_keras3, tpr_keras3, thresholds_keras3 = roc_curve(y, y_pred_keras3.ravel())
    auc_keras3 = auc(fpr_keras3, tpr_keras3)
    ax = fig.add_subplot(2, 2, plt_index)
    axins = ax.inset_axes((0.5, 0.4, 0.5, 0.4))
    ax.plot(fpr_keras, tpr_keras, color=colors[0], linestyle=':',
            label='semi & ' + 'K=' + repr(K) + ' (area = {:.3f})'.format(auc_keras))
    ax.plot(fpr_keras2, tpr_keras2, color=colors[1], linestyle='-.',
            label='supervised & ' + 'K=' + repr(K) + ' (area = {:.3f})'.format(auc_keras2))
    ax.plot(fpr_keras3, tpr_keras3, color=colors[2], label='supervised (area = {:.3f})'.format(auc_keras3))
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title('l=' + repr(number1) + ', ' + 'u=' + repr(number2))
    ax.legend(loc="lower left")
    axins.plot(fpr_keras, tpr_keras, color=colors[0], linestyle=':')
    axins.plot(fpr_keras2, tpr_keras2, color=colors[1], linestyle='-.')
    axins.plot(fpr_keras3, tpr_keras3, color=colors[2])
    zone_left = 0
    zone_right = xlims_right[plt_index - 1]
    ylim0 = ylims[plt_index - 1]
    ylim1 = 0.99
    # 调整子坐标系的显示范围
    axins.set_xlim(zone_left, zone_right)
    axins.set_ylim(ylim0, ylim1)

plt.tight_layout(pad=1, h_pad=3.0, w_pad=3.0)
plt.show()