import arff
import bz2
import pickle
from scipy import sparse

class Dataset(object):
    @classmethod
    def load_arff_to_numpy(cls, filename, labelcount, endian = "big", input_feature_type = 'float', encode_nominal = True, load_sparse = False):
        matrix = None
        if not load_sparse:
            arff_frame = arff.load(open(filename,'rb'), encode_nominal = encode_nominal, return_type=arff.DENSE)
            matrix = sparse.csr_matrix(arff_frame['data'], dtype=input_feature_type)
        else:
            arff_frame = arff.load(open(filename ,'rb'), encode_nominal = encode_nominal, return_type=arff.COO)
            data = arff_frame['data'][0]
            row  = arff_frame['data'][1]
            col  = arff_frame['data'][2]
            matrix = sparse.coo_matrix((data, (row, col)), shape=(max(row)+1, max(col)+1))

        X, y = None, None
        
        if endian == "big":
            X, y = matrix.tocsc()[:,labelcount:].tolil(), matrix.tocsc()[:,:labelcount].astype(int).tolil()
        elif endian == "little":
            X, y = matrix.tocsc()[:,:-labelcount].tolil(), matrix.tocsc()[:,-labelcount:].astype(int).tolil()
        else:
            return None

        return X, y

    @classmethod
    def save_to_arff(cls, X, y, endian = "big", save_sparse = False):
        X = X.todok()
        y = y.todok()
        
        x_prefix = 0
        y_prefix = 0

        x_attributes = [(u'X{}'.format(i),u'NUMERIC') for i in xrange(X.shape[1])]
        y_attributes = [(u'y{}'.format(i), [unicode(0),unicode(1)]) for i in xrange(y.shape[1])]

        if endian == "big":
            y_prefix = X.shape[1]
            relation_sign = -1
            attributes = x_attributes + y_attributes

        elif endian == "little":
            x_prefix = y.shape[1]
            relation_sign = 1
            attributes = y_attributes + x_attributes 

        else:
            raise ValueError("Endian not in {big, little}")

        if save_sparse:
            data = [{} for r in xrange(X.shape[0])]
        else:
            data = [[0 for c in xrange(X.shape[1] + y.shape[1])] for r in xrange(X.shape[0])]
        
        for keys, value in X.iteritems():
            data[keys[0]][x_prefix + keys[1]] = value

        for keys, value in y.iteritems():
            data[keys[0]][y_prefix + keys[1]] = value

        dataset = {
            u'description': u'traindata',
            u'relation': u'traindata: -C {}'.format(y.shape[1] * relation_sign),
            u'attributes': attributes,                
            u'data': data
        }

        return arff.dumps(dataset)

    @classmethod
    def save_dataset_dump(cls, filename, input_space, labels):
        if filename[-4:] != '.bz2':
            filename += ".bz2"

        with bz2.BZ2File(filename, "wb") as file_handle:
            pickle.dump({'X': input_space, 'y': labels}, file_handle)

    @classmethod
    def load_dataset_dump(cls, filename):
        data = None

        if filename[-4:] != '.bz2':
            filename += ".bz2"

        with bz2.BZ2File(filename, "r") as file_handle:
            data = pickle.load(file_handle)
        
        return data