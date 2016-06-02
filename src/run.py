# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import dataset
import mlknn


data_dir = '/home/yw/tmp/yelp/data/'
feature_num = 64
label_num = 9
thresh = 0.44


def generate_train_set():
    df2 = pd.read_csv(data_dir + 'train_all.txt.1000', sep=' ')
    df = df2[df2['labels'].notnull()]
    
    row_num = len(df)
    cols = ['F%d' % (i+1) for i in xrange(feature_num)]
    
    y = np.zeros((row_num, label_num), dtype='int')
    X = np.asarray(df[cols].values, dtype='float64')
    
    count = 0    
    for _, row in df.iterrows():
        labels = str(row['labels'])
        label_ids = [int(l) for l in labels.split(',')]
        for l in label_ids:
            y[count][l] =1
        count += 1
        
    dataset.Dataset.save_dataset_dump(data_dir + 'yelp_train', X, y)


def generate_test_set():
    df = pd.read_csv(data_dir + 'test_all.txt.1000', sep=' ')
    
    row_num = len(df)
    
    cols = ['F%d' % (i+1) for i in xrange(feature_num)]
    
    y = np.zeros((row_num, label_num), dtype='int')
    X = np.asarray(df[cols].values, dtype='float64')
        
    dataset.Dataset.save_dataset_dump(data_dir + 'yelp_test', X, y)
        
    
def classify_image():
    df = pd.read_csv(data_dir + 'test_all.txt.1000', sep=' ')
    
    cols = ['F%d' % (i+1) for i in xrange(feature_num)]
    
    X = np.asarray(df[cols].values, dtype='float64')
    
    train_set = dataset.Dataset.load_dataset_dump(data_dir + 'yelp_train')
    
    clf = mlknn.KNearestNeighbours()
    clf.fit(train_set['X'], train_set['y'])
    
    result = clf.predict(X)
    np.savetxt(data_dir + 'result.txt', result, fmt='%d')


def classify_business():
    df = pd.read_csv(data_dir + 'test_all.txt.1000', sep=' ')
    res = np.loadtxt(data_dir + 'result.txt', dtype='int')
    
    bizs = dict()
    
    for idx, row in df.iterrows():
        if str(row['business_id']) in bizs:
            bizs[str(row['business_id'])] = np.vstack([bizs[str(row['business_id'])], res[idx]])
        else:
            bizs[str(row['business_id'])] = np.array([res[idx]])
    
    new_bizs = dict().fromkeys(bizs.keys())
    
    for k, v in bizs.iteritems():
        print np.mean(v, axis=0)
        pred = np.mean(v, axis=0) > thresh
        nz = pred.nonzero()
        new_bizs[k] = [str(x) for x in nz[0]]

    outf = open(data_dir + 'res.csv', 'w')
    print >> outf, 'business_id,labels'
    
    ids = pd.read_csv(data_dir + 'sample_submission.csv').values[:, 0]
    for biz_id in ids:
        if str(biz_id) in new_bizs:
            print >> outf, str(biz_id) + ',' + ' '.join(new_bizs[str(biz_id)])
        else:
            print >> outf, str(biz_id) + ','
    
    outf.close()


if __name__ == "__main__":
    print 'generating train set...\n'
    generate_train_set()
    print 'generating test set...\n'
    generate_test_set()
    print 'classifying each image...\n'
    classify_image()
    print 'classifying each business...\n'
    classify_business()

    
        
    