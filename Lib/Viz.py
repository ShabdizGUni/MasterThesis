import itertools
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45, fontsize=2)
#     plt.yticks(tick_marks, classes, fontsize=2)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  fontsize=2,
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')


def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_accuracy_dev(acc, val_acc, filepath, title=None):
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.figure()
    sns.set_style('darkgrid')
    plt.plot(acc)
    plt.plot(val_acc)
    plt.ylabel('accuray')
    plt.xlabel('epochs')
    title_label = 'Accuracy over Epochs' if title is None else 'Accuracy over Epochs :' + title
    plt.title(title_label)
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filepath)


def plot_loss_dev(loss, val_loss, filepath, title=None):
    plt.clf()
    plt.figure()
    sns.set_style('darkgrid')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    title_label = 'Loss over Epochs' if title is None else 'Loss over Epochs: ' + title
    plt.title(title_label)
    plt.legend(['train', 'test'], loc='lower left')
    plt.savefig(filepath)

