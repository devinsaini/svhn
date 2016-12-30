import h5py
import numpy as np
from scipy import misc

def read_process_h5(filename):
    """ Reads and processes the mat files provided in the SVHN dataset. 
        Input: filename 
        Ouptut: list of python dictionaries 
    """
    
    f = h5py.File(filename, 'r')
    groups = f['digitStruct'].items()
    bbox_ds = np.array(groups[0][1]).squeeze()
    names_ds = np.array(groups[1][1]).squeeze()
 
    data_list = []
    num_files = bbox_ds.shape[0]
    count = 0
 
    for objref1, objref2 in zip(bbox_ds, names_ds):
 
        data_dict = {}
 
        # Extract image name
        names_ds = np.array(f[objref2]).squeeze()
        filename = ''.join(chr(x) for x in names_ds)
        data_dict['filename'] = filename
 
        #print filename
 
        # Extract other properties
        items1 = f[objref1].items()
 
        # Extract image label
        labels_ds = np.array(items1[1][1]).squeeze()
        try:
            label_vals = [int(f[ref][:][0, 0]) for ref in labels_ds]
        except TypeError:
            label_vals = [labels_ds]
        data_dict['labels'] = label_vals
        data_dict['length'] = len(label_vals)
 
        # Extract image height
        height_ds = np.array(items1[0][1]).squeeze()
        try:
            height_vals = [f[ref][:][0, 0] for ref in height_ds]
        except TypeError:
            height_vals = [height_ds]
        data_dict['height'] = height_vals
 
        # Extract image left coords
        left_ds = np.array(items1[2][1]).squeeze()
        try:
            left_vals = [f[ref][:][0, 0] for ref in left_ds]
        except TypeError:
            left_vals = [left_ds]
        data_dict['left'] = left_vals
 
        # Extract image top coords
        top_ds = np.array(items1[3][1]).squeeze()
        try:
            top_vals = [f[ref][:][0, 0] for ref in top_ds]
        except TypeError:
            top_vals = [top_ds]
        data_dict['top'] = top_vals
 
        # Extract image width
        width_ds = np.array(items1[4][1]).squeeze()
        try:
            width_vals = [f[ref][:][0, 0] for ref in width_ds]
        except TypeError:
            width_vals = [width_ds]
        data_dict['width'] = width_vals
 
        data_list.append(data_dict)
 
        count += 1
        #print ('Processed: {}, {}'.format(count, num_files))
 
    return data_list
    
    
def createImageandLabelData(datapoint, output_size, path):
    rawimg = misc.imread(path+datapoint['filename'])
    img = np.array(misc.imresize(rawimg, size=output_size, interp='bilinear'))
    labels = datapoint['labels']
    
    return img, labels

def createImageData(datapoint, output_size, path):
    rawimg = misc.imread(path+datapoint['filename'])
    img = np.array(misc.imresize(rawimg, size=output_size, interp='bilinear'))
    return img, rawimg.shape

def createRelativeCoordinates(datapoint, idx, imsize):
    top = datapoint['top'][idx]/imsize[0]
    left= datapoint['left'][idx]/imsize[1]
    height = datapoint['height'][idx]/imsize[0]
    width = datapoint['width'][idx]/imsize[1]
    
    return (top, left, height, width)

def createCoordinates(datapoint, idx, imsize, output_size):
    top = datapoint['top'][idx]/imsize[0]*output_size[0]
    left= datapoint['left'][idx]/imsize[1]*output_size[1]
    height = datapoint['height'][idx]/imsize[0]*output_size[0]
    width = datapoint['width'][idx]/imsize[1]*output_size[1]
    
    return (top, left, height, width)