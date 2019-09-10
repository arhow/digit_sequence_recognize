import bcolz


def save_array(data, fname):
    print("Saving image dataset at the location " + str(fname) + ".")
    c = bcolz.carray(data, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    print("Loading image dataset from the location " + str(fname) + ".")
    return bcolz.open(fname)[:]