from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

class Metrics:
    def __init__(self, y_true, y_predict, labels):
        self.y_true = y_true
        self.y_predict = y_predict
        self.labels = labels

        #term added to the denominator to improve numerical stability. avoid dividing by zero
        self.eps = 1e-6

    def confusion_matrix(self):
        return metrics.confusion_matrix(self.y_true, self.y_predict, self.labels)

    def balanced_score(self):
        cm = self.confusion_matrix()
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize
        val = 0.0
        for i in range(cm.shape[0]):
            val += cm[i, i]

        return val/cm.shape[0]

    def represent_cm(self):
        cm = self.confusion_matrix()
        # visualize normalized confusion Matrix
        np.set_printoptions(precision=2)

        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=self.labels, yticklabels=self.labels,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return plt


