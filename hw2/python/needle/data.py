import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.flip(img, axis=1)
        else :
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        ret = np.roll(img, (-shift_x, -shift_y) , (0,1))

        X = img.shape[0]
        Y = img.shape[1]

        arange_x = np.arange(X-shift_x,X) if shift_x > 0 else np.arange(0-shift_x)
        arange_y = np.arange(Y-shift_y,Y) if shift_y > 0 else np.arange(0-shift_y)

        ret[arange_x[:X] , : , :] = 0
        ret[: , arange_y[:Y] , :] = 0

        return ret
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

import gzip
class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        print(image_filename)
        print(label_filename)
        self.imgs , self.labels = self.parse_mnist(image_filename, label_filename)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img , label = (self.imgs[index], self.labels[index])
        if self.transforms:
            for t in self.transforms:
                img = t(img)
        return (img, label)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION

    def parse_mnist(self, image_filename, label_filename):
        """ Read an images and labels file in MNIST format.  See this page:
        http://yann.lecun.com/exdb/mnist/ for a description of the file format.
        Args:
            image_filename (str): name of gzipped images file in MNIST format
            label_filename (str): name of gzipped labels file in MNIST format
        Returns:
            Tuple (X,y):
                X (numpy.ndarray[np.float32]): 4D numpy array containing the loaded 
                    data.  The dimensionality of the data should be 
                    (num_examples x H x W x C) where 'input_dim' is the full 
                    dimension of the data, e.g., since MNIST images are 28x28, it 
                    will be 28x28x1.  Values should be of type np.float32, and the data 
                    should be normalized to have a minimum value of 0.0 and a 
                    maximum value of 1.0. The normalization should be applied uniformly
                    across the whole dataset, _not_ individual images.
                y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                    labels of the examples.  Values should be of type np.uint8 and
                    for MNIST will contain the values 0-9.
        """
        ### BEGIN YOUR CODE
        with gzip.open(image_filename) as image_file:
          pixels = np.frombuffer(image_file.read(), 'B', offset=16)
        images = pixels.reshape(-1, 28, 28 ,1).astype('float32') / 255

        with gzip.open(label_filename) as f:
          # First 8 bytes are magic_number, n_labels
          integer_labels = np.frombuffer(f.read(), 'B', offset=8)
        labels = integer_labels.astype('uint8')

        return (images, labels)
        ### END YOUR CODE

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
