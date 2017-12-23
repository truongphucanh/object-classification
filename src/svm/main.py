import numpy as np
import cv2, os, glob, pickle, logging, time, ulti
import _pickle as cPickle
from scipy.cluster.vq import vq 
from sklearn import svm, metrics
from algorithms import kernels, multi_classifier

def config():
    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(linewidth=300)

def cal_histogram_of_words(image, vocabulary):
   """
   Calculate histogram of words for the image as its feature using for svm

   Parameters:
   -----------
   image : numpy array (2D)
       the gray scale image

   vocabulary : numpy array (2D)
       each row is a words (128D SIFT vector, center of cluster when running kmeans for bow)

   Returns:
   --------
   hist : numpy array
       histogram of words for the image
   """

   n_words = vocabulary.shape[0]
   hist = np.zeros(n_words)
   sift = cv2.xfeatures2d.SIFT_create()
   kpoints, descriptors = sift.detectAndCompute(image,None)
   words, distance = vq(descriptors,vocabulary)
   for word in words:
       hist[word] += 1
   return hist

def cal_features_for_images(directory):
    """
    Calculate features for all images in directory

    Feature is histogram of words.

    Vocabulary is loaded from "../bow/vocabulary.pkl".

    Vocabulary has 500 words.

    n_images from test folder is 318 image.

    X_test : features of test images, saved in "features/X_test.pkl"
    y_test : labels of test images, saved in "features/y_test.pkl"
    filenames will be saved in "src/log/filenames.log" for access after run svm

    Returns:
    --------
    0 if success
    """

    X_test = []
    y_test = []
    log_filenames = open('log\\filenames.log', 'w')
    with open('bin\\vocabulary.pkl', 'rb') as fid:
        vocabulary = pickle.load(fid)
    print('n_words: {}'.format(vocabulary.shape[0]))
    for folder in glob.glob(directory):
        drive, path = os.path.splitdrive(folder)
        path, label = os.path.split(path)
        for filename in glob.glob(folder+'/*.*'):
            print('calculating for imgage {}'.format(filename))
            im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            hist = cal_histogram_of_words(im, vocabulary)
            X_test.append(hist)
            y_test.append(label)
            log_filenames.write('{} \n'.format(filename))
    with open('features\\X_test.pkl', 'wb') as fp:
        pickle.dump(X_test, fp)
    with open('features\\y_test.pkl', 'wb') as fp:
        pickle.dump(y_test, fp)
    return 0

if __name__ == '__main__':
    config()
    print('Loading X_train, y_train')
    with open('features\\X_train.pkl', 'rb') as fid:
        X_train = pickle.load(fid)
    with open('features\\y_train.pkl', 'rb') as fid:
        y_train = pickle.load(fid)

    #cal_features_for_images('..\\datasets\\test\\*')
    with open('features\\X_test.pkl', 'rb') as fid:
        X_test = pickle.load(fid)
    with open('features\\y_test.pkl', 'rb') as fid:
        y_test = pickle.load(fid)

    print('X_train: ', np.shape(X_train))
    print('y_train: ', np.shape(y_train))
    print('X_test: ', np.shape(X_test))
    print('y_test: ', np.shape(y_test))

    # C = 1000
    # kernel = 'linear'
    # gamma = None
    # degree = None
    # multi_class = 'ovr'
    # model = svm.LinearSVC(C = C)
        
    C = 1
    kernel = 'linear'
    gamma = None
    degree = None
    multi_class = 'ovo'
    model = svm.SVC(kernel = kernel, gamma = gamma, C = C)

    # C = 1000
    # kernel = 'rbf'
    # gamma = 1e-06
    # degree = None
    # multi_class = 'ovo'
    # model = svm.SVC(kernel = kernel, gamma = gamma, C = C)

    # C = 0.1
    # kernel = 'poly'
    # gamma = None
    # degree = 3
    # multi_class = 'ovo'
    # model = svm.SVC(kernel = kernel, degree=degree, C=C)

    #kernel = kernels.LinearKernel()
    #model = multi_classifier.OneVsRestClassifier()
    #C = 1.0

    logger = ulti.create_logger('log\\svm\\{}_{}_{}_{}_{}.log'.format(multi_class, kernel, C, gamma, degree), logging.DEBUG, logging.DEBUG)
    logger.info('*'*75)
    logger.info('C: {}'.format(C))
    logger.info('multi_class: {}'.format(multi_class))
    logger.info('kernel: {}'.format(kernel))
    logger.info('gamma: {}'.format(gamma))
    logger.info('degree: {}'.format(degree))

    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time
    logger.info('fit_time: {} seconds'.format(fit_time))
    model_file = 'bin\\svm-models\\{}_{}_{}_{}_{}.pkl'.format(multi_class, kernel, C, gamma, degree)
    with open(model_file, 'wb') as fid:
        cPickle.dump(model, fid) 
    logger.info('model is saved in {}'.format(model_file))

    y_pred_train = model.predict(X_train)
    #logger.info('classification report on train set:\n {}'.format(metrics.classification_report(y_train, y_pred_train)))
    logger.info('score on train set: {}'.format(model.score(X_train, y_train)))

    y_pred_test = model.predict(X_test)
    #logger.info('classification report on test set:\n {}'.format(metrics.classification_report(y_test, y_pred_test)))
    logger.info('score on test set: {}'.format(model.score(X_test, y_test)))
    logger.info('confusion_matrix:\n {}'.format(metrics.confusion_matrix(y_test, y_pred_test, np.unique(y_test))))
    logger.info('%-15s\t%-15s' % ('y_test', 'y_pred_test'))
    for idx in range(len(y_test)):
        logger.info('%-15s\t%-15s' % (y_test[idx], y_pred_test[idx]))
