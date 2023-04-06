import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
import warnings
from sklearn import metrics
from model import Train
# train_fine1
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    auc_result, fprs, tprs, accuracy_test, precision_test, recall_test, f1_test = Train()
   
    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(accuracy_test), np.std(accuracy_test)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(precision_test), np.std(precision_test)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_test), np.std(recall_test)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_test), np.std(f1_test)))

    tpr = []
    mean_fpr = np.linspace(0, 1, 10000)
    for i in range(len(fprs)):
        np.savetxt('model_train/seqVec/fprs_%s.txt' %(str(i)), fprs[i], fmt='%f', delimiter=',')
        np.savetxt('model_train/seqVec/tprs_%s.txt' %(str(i)), tprs[i], fmt='%f', delimiter=',')
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, label='ROC  (AUC = %.4f)' % ( auc_result[i]),color = 'b')
    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    auc_std = np.std(auc_result)
    plt.plot(mean_fpr, mean_tpr, color='b', alpha=0.8, label='Mean AUC (AUC = %.4f $\pm$ %.4f)' % (mean_auc, auc_std))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')     
    plt.legend(loc='lower right')
    plt.savefig('piture/mul_test1')
    plt.show()

