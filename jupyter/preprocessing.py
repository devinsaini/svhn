import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator

class SVHNImageDataGenerator(ImageDataGenerator):
	def standardize(self, x):
		if self.preprocessing_function:
			x = self.preprocessing_function(x)
		if self.rescale:
			x *= self.rescale
		# x is a single image, so it doesn't have image number at index 0
		img_channel_index = self.channel_index - 1
        
		# fix start
		img_row_index = self.row_index - 1
		img_col_index = self.col_index - 1
		img_channel_index = self.channel_index - 1
		if self.samplewise_center:
			x -= np.mean(x, axis=(img_channel_index, img_row_index, img_col_index), keepdims=True)
		if self.samplewise_std_normalization:
			x /= (np.std(x, axis=(img_channel_index, img_row_index, img_col_index), keepdims=True) + 1e-7)
		# fix end

		if self.featurewise_center:
			if self.mean is not None:
				x -= self.mean
			else:
				warnings.warn('This ImageDataGenerator specifies '
							'`featurewise_center`, but it hasn\'t'
							'been fit on any training data. Fit it '
							'first by calling `.fit(numpy_data)`.')
		if self.featurewise_std_normalization:
			if self.std is not None:
				x /= (self.std + 1e-7)
			else:
				warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
		if self.zca_whitening:
			if self.principal_components is not None:
				flatx = np.reshape(x, (x.size))
				whitex = np.dot(flatx, self.principal_components)
				x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
			else:
				warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
		return x
        
	def flow(self, Ximg, Xidx, ycount, ylabel, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
		return SVHNNumpyArrayIterator(
            Ximg, Xidx, ycount, ylabel, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


            

class SVHNNumpyArrayIterator(Iterator):

    def __init__(self, Ximg, Xidx, ycount, ylabel, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg'):

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.Ximg = np.asarray(Ximg)
        self.Xidx = np.asarray(Xidx)
        self.ycount = np.asarray(ycount)
        self.ylabel = np.asarray(ylabel)
        if self.Ximg.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.X.shape)
        channels_axis = 3 if dim_ordering == 'tf' else 1
        if self.Ximg.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('NumpyArrayIterator is set to use the '
                             'dimension ordering convention "' + dim_ordering + '" '
                             '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                             'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                             'However, it was passed an array with shape ' + str(self.Ximg.shape) +
                             ' (' + str(self.Ximg.shape[channels_axis]) + ' channels).')
        
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(SVHNNumpyArrayIterator, self).__init__(Ximg.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_ximg = np.zeros(tuple([current_batch_size] + list(self.Ximg.shape)[1:]))
        for i, j in enumerate(index_array):
            ximg = self.Ximg[j]
            ximg = self.image_data_generator.random_transform(ximg.astype('float32'))
            ximg = self.image_data_generator.standardize(ximg)
            batch_ximg[i] = ximg

        batch_ycount = self.ycount[index_array]
        batch_ylabel = self.ylabel[index_array]
        batch_Xidx = self.Xidx[index_array]
        return [batch_ximg, batch_Xidx], [batch_ycount, batch_ylabel]
	