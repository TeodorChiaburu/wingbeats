"""Library for postprocessing functions"""



# Import libraries
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy import signal



def mean_plot(X, title = 'Mean amplitudes', figsize = None, path_to_save = None):
    """Plot mean amplitudes of signals in **X**.
    
    :param X: Matrix of signals.
    :type X: list or array
    :param title: Title of the plot. Defaults to 'Mean amplitudes'.
    :type title: str
    :param figsize: Figure size. Defaults to *None*.
    :type figsize: tuple, optional
    :param path_to_save: Location for saving the plot as a png. Defaults to *None*.
    :type path_to_save: str, optional
    :return: Array of mean amplitudes
    """
  
    mean_X = np.mean(X, axis=0)
    if figsize is not None:
        plt.figure(figsize = figsize)
    plt.plot(np.linspace(0, len(X[0]), len(X[0])), mean_X)
    plt.title(title)
    plt.grid()
  
    if path_to_save is not None:
        plt.savefig(path_to_save + title + '.png', bbox_inches = 'tight')
    
    return mean_X
    
#############################################################################################
    
def plot_accuracy(history, title = 'Accuracy', figsize = None, fontsize = 15, 
                  show = True, path_to_save = None):
    """Plot training and validation accuracy curves. 
    
    :param history: Dictionary of training statistics.
    :type history: dict
    :param title: Title of the plot. Defaults to 'Accuracy'.
    :type title: str
    :param figsize: Figure size. Defaults to *None*.
    :type figsize: tuple, optional
    :param fontsize: Fontsize in legend. Defaults to 15.
    :type fontsize: int
    :param show: Whether to show the plot. Defaults to *True*.
    :type show: bool
    :param path_to_save: Location for saving the plot as a png. Defaults to *None*.
    :type path_to_save: str, optional
    """
    
    if figsize is not None:
        plt.figure(figsize = figsize)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'lower right', fontsize = fontsize)
    plt.grid()
    if path_to_save is not None:
        plt.savefig(path_to_save + title + '.png', bbox_inches = 'tight')
    if show:
        plt.show()
    
#############################################################################################
    
def plot_all_transforms(sig, fs, cutoff, nperseg, noverlap, 
                        title_size = 32, num_rows = 1, row_ind = 0):
    """Plot a raw signal along with its FFT, PSD and spectrogram.
    
    :param sig: Raw signal.
    :type sig: list or array
    :param fs: Sampling frequency.
    :type fs: int
    :param cutoff: Number of frequencies to keep in FFT, PSD and spectrograms.
    :type cutoff: int
    :param nperseg: Length of a segment to apply the Welch Transform to.
    :type nperseg: int
    :param noverlap: Lenth of overlapping region between segments.
    :type noverlap: int
    :param title_size: Size of the plot titles. Defaults to 32.
    :type title_size: int
    :param num_rows: Number of rows (in case multiple signals are plotted). Defaults to 1. 
    :type num_rows: int
    :param row_ind: Index of current row (in case multiple signals are plotted). Defaults to 0.
    :type row_ind: int
    """
    
    len_sig = len(sig)
    duration = len_sig/fs
    cutoff_hz = int( fs / nperseg * (cutoff-1) ) # conversion into HZ
    label_size = ticks_size = title_size - 5

    # Raw
    plt.subplot(num_rows, 4, 4*row_ind+1)
    plt.plot(np.linspace(0, duration, len_sig), sig)
    plt.title('Raw', fontsize = title_size)
    plt.xlabel('Time (s)', fontsize = label_size)
    plt.xticks(np.arange(0., duration, step=0.05), fontsize = ticks_size)
    plt.yticks(np.arange(-1., 1.01, step=0.5), fontsize = ticks_size)
    plt.grid()

    # FFT
    four = np.fft.fft(sig)
    four = np.abs(four[:len_sig*cutoff_hz//fs])
    four_max = np.max(four)
    plt.subplot(num_rows, 4, 4*row_ind+2)
    plt.plot(np.linspace(0, cutoff_hz, len_sig*cutoff_hz//fs), four)
    plt.title('FFT', fontsize = title_size)
    plt.xlabel('Frequency [kHz]', fontsize = label_size)
    plt.yticks(np.arange(0, four_max, step = four_max//4), fontsize = ticks_size)
    plt.xticks(np.arange(0., cutoff_hz+0.1, step=1000), labels = list(range(cutoff_hz//1000 + 1)), 
               fontsize = ticks_size)
    plt.grid()

    # PSD
    filtered_psd = 10*np.log10(signal.welch(sig, fs = fs, window = 'hanning', 
                                            nperseg = nperseg, noverlap = noverlap)[1])[:cutoff]
    psd_min, psd_max = np.min(filtered_psd), np.max(filtered_psd)
    plt.subplot(num_rows, 4, 4*row_ind+3)
    plt.plot(np.linspace(0, cutoff_hz, cutoff), filtered_psd)
    plt.title('PSD', fontsize = title_size)
    plt.xlabel('Frequency [kHz]', fontsize = label_size)
    plt.ylabel('PSD [dB]', fontsize = label_size)
    plt.xticks(np.arange(0., cutoff_hz+0.1, step=1000), labels = list(range(cutoff_hz//1000 + 1)), 
               fontsize = ticks_size)
    plt.yticks(np.arange(int(psd_min), int(psd_max), step = (psd_max-psd_min)//4), fontsize = ticks_size)
    plt.grid()

    # Spectrogram
    plt.subplot(num_rows, 4, 4*row_ind+4)
    plt.specgram(sig, NFFT = nperseg, Fs = fs, noverlap = noverlap, xextent = (0.0, duration),
                 vmin=-110, vmax=-30)
    plt.title('Spectrogram', fontsize = title_size)
    plt.xlabel('Time (s)', fontsize = label_size)
    plt.ylabel('Frequency [kHz]', fontsize = label_size)
    plt.xticks(np.arange(0.05, duration, step=0.05), fontsize = ticks_size)
    plt.yticks(np.arange(0, cutoff_hz+0.1, step=1000), labels = list(range(cutoff_hz//1000 + 1)),
              fontsize = ticks_size)
    plt.ylim(top = cutoff_hz)
    
#############################################################################################

def plot_heatmap(mat, xticklab, yticklab, xlab = 'x', ylab = 'y', title = 'Heat map',
                 mask = None, path_to_save = None, title_size = 15, ticks_size = 15, 
                 label_size = 15, item_size = 15, cbar = False):
    """Plot matrix as a heat map.
    
    :param mat: Matrix to plot.
    :type mat: array
    :param xticklab: X ticks labels.
    :type xticklab: str
    :param yticklab: Y ticks labels.
    :type yticklab: str
    :param xlab: X label. Defaults to 'x'.
    :type xlab: str
    :param ylab: Y label. Defaults to 'y'.
    :type ylab: str
    :param title: Plot title. Defaults to 'Heat map'.
    :type title: str
    :param mask: Matrix of 1's and 0's specifying which entries of the matrix to mask. Defaults to *None*.
    :type mask: array, optional
    :param path_to_save: Location for saving the plot as a png. Defaults to *None*.
    :type path_to_save: str, optional
    :param title_size: Size of the plot titles. Defaults to 15.
    :type title_size: int
    :param ticks_size: Size of the axis ticks. Defaults to 15.
    :type ticks_size: int
    :param label_size: Size of the axis labels. Defaults to 15.
    :type label_size: int
    :param item_size: Size of each matrix entry. Defaults to 15.
    :type item_size: int
    """
    
    sns.heatmap(mat, annot = True, cmap = plt.cm.Blues, mask = mask,
                xticklabels = xticklab, yticklabels = yticklab, cbar = cbar, square = True, 
                annot_kws = {"size": item_size}, linewidths = 1, linecolor = 'black')
    plt.xticks(rotation = 45, ha = "right", rotation_mode = "anchor", fontsize = ticks_size)
    plt.yticks(rotation = 'horizontal', fontsize = ticks_size)
    plt.title(title, fontsize = title_size)
    plt.xlabel(xlab, fontsize = label_size)
    plt.ylabel(ylab, fontsize = label_size)   
    if path_to_save is not None:
        plt.savefig(path_to_save + title + '.png', bbox_inches = 'tight')
        
#############################################################################################
  
def plot_confusion(y_true, y_pred, title = 'Confusion matrix', 
                   axis_labels = None, hide_diag = False, path_to_save = None):
    """Compute and plot confusion matrix. 
    
    :param y_true: True labels.
    :type y_true: list or array
    :param y_pred: Predicted labels.
    :type y_pred: list or array
    :param title: Plot title. Defaults to 'Confusion matrix'.
    :type title: str
    :param axis_labels: X and y labels. Defaults to *None*.
    :type axis_labels: str, optional
    :param hide_diag: Whether to mask the diagonal entries. Defaults to *False*.
    :type hide_diag: bool
    :param path_to_save: Location for saving the plot as a png. Defaults to *None*.
    :type path_to_save: str, optional
    """
    
    # Build the confusion matrix
    conf_mat = np.round( confusion_matrix(y_true, y_pred, normalize = 'true'), 2 )
    
    # If the user wishes to emphasize the misclassifications, 
    # the diagonal accuracies can be leaved out of the heat map
    if hide_diag:
        mask = np.zeros_like(conf_mat)
        mask[np.diag_indices_from(mask)] = True
    else:
        mask = None
    
    # Plot confusion matrix along with a heatmap
    plt.figure(figsize=(8, 8))
    plot_heatmap(conf_mat, xticklab = axis_labels, yticklab = axis_labels, 
                 xlab = 'Predicted label', ylab = 'True label', title = title,
                 mask = mask, path_to_save = path_to_save)
    plt.tight_layout()
    plt.show()
    
#############################################################################################
    
def compute_mean_conf_mat(conf_mats):
    """Compute means and standard deviations itemwise for a list of (confusion) matrices.
    
    :param conf_mats: Matrices to compute the statistics on.
    :type conf_mats: list
    :return: Mean and standard deviation matrices (rounded to 0.001).
    :rtype: tuple
    """
    
    dim = conf_mats[0].shape[0] # dimension of genus conf. mat.
    conf_mats = np.asarray(conf_mats)    
    mean_mat, std_mat = np.zeros((dim, dim)), np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            mean_mat[i, j] = np.mean(conf_mats[:, i, j])
            std_mat[i, j]  = np.std(conf_mats[:, i, j])
            
    return np.round(mean_mat, 3), np.round(std_mat, 3)
    
#############################################################################################

def plot_roc(y_true, y_pred, labels, title = 'ROC', 
             title_size = 15, ticks_size = 15, label_size = 15):
    """Plot multi-class ROC along with the individual AUC values.
    
    :param y_true: True labels (needs to be a binarized vector).
    :type y_true: array
    :param y_pred: Predicted labels.
    :type y_pred: array.
    :param labels: Names of classes.
    :type labels: list
    :param title: Plot title. Defaults to 'ROC'.
    :type title: str
    :param title_size: Size of the plot titles. Defaults to 15.
    :type title_size: int
    :param ticks_size: Size of the axis ticks. Defaults to 15.
    :type ticks_size: int
    :param label_size: Size of the axis labels. Defaults to 15.
    :type label_size: int
    """
    
    n_classes = len(labels)
    
    # Compute ROC curve for every class + area under curve
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw = 3,
                 label = 'AUC {0} = {1:0.2f}'
                 ''.format(labels[i], roc_auc))
    
    plt.plot([0, 1], [0, 1], 'k--', lw = 3) # y=x line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = label_size)
    plt.ylabel('True Positive Rate', fontsize = label_size)
    plt.xticks(fontsize = ticks_size)
    plt.yticks(fontsize = ticks_size)
    plt.title(title, fontsize = title_size)
    leg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc = 'upper left', fontsize = label_size)
    for line in leg.get_lines():
      line.set_linewidth(4.0)
      
#############################################################################################
      
def autolabel(rects, coords = (0, 0), label_size = 15):
    """Attach a text label above each bar in **rects**, displaying its height.
    
    :param rects: Bars from barplot.
    :type rects: matplotlib object
    :param coords: Label coordinates. Defaults to (0, 0).
    :type coords: tuple
    :param label_size: Size of the labels above the bars. Defaults to 15.
    :type label_size: int
    """

    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=coords,  # text offset
                    textcoords="offset points",
                    ha='right', va='center', fontsize = label_size,
                    rotation = 90, rotation_mode = "anchor") 
        
        
