from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
from tract_feat import _feat_to_3D
from fiber_distance import fiber_pair_similarity

class Fiber(data.Dataset):
    def __init__(self, ds,transform=None):
        self.ds =ds
        #self.dsf = dsf
        self._cache = dict()
        self.transform = transform

    def __getitem__(self, index: int):
        img=self.ds[index]
        img=np.expand_dims(img, axis=0)
        #imgf = self.dsf[index]
        img=torch.tensor(img, dtype=torch.float)
        #imgf=torch.tensor(imgf, dtype=torch.float)
        # if self.transform is not None:
        #     img = self.transform(img)
        return tuple(img) #imgf

    def __len__(self) -> int:
        return len(self.ds)


class Fiber_pair(data.Dataset):
    def __init__(self,vec,roi,surf,transform=None):
        #vec=np.reshape(vec,(len(ds),-1,ds.shape[1]))
        self.vec=vec #8000*14*3
        self.roi=roi
        self.transform = transform
        self.surf=surf

    def __getitem__(self, index: int):
        index1=index
        index2=np.random.randint(0,len(self.vec))
        # img1=self.ds[index1]
        # img2 = self.ds[index2]
        fiber1=self.vec[index1]
        fiber2 = self.vec[index2]
        # fiber_norm1=self.x_norm[index1]
        # fiber_norm2 = self.x_norm[index2]
        # img1= _feat_to_3D(np.expand_dims(fiber1,0), repeat_time=14).squeeze()
        # img1=img1.transpose(2,0,1)
        # img2= _feat_to_3D(np.expand_dims(fiber2,0), repeat_time=14).squeeze()
        # img2=img2.transpose(2,0,1)
        roi1=self.roi[index1]
        #roi2 = self.roi[index2]
        surf1=self.surf[index1]
        similarity=fiber_pair_similarity(fiber1, fiber2)
        # img1=torch.tensor(img1, dtype=torch.float)
        # img2 = torch.tensor(img2, dtype=torch.float)
        fiber1=torch.tensor(fiber1.T, dtype=torch.float)
        fiber2 = torch.tensor(fiber2.T, dtype=torch.float)
        similarity = torch.tensor(similarity, dtype=torch.float)
        # if self.transform is not None:
        #     img = self.transform(img)
        return fiber1,fiber2,similarity,roi1,surf1,index

    def __len__(self) -> int:
        return len(self.vec)
class FiberMap_pair(data.Dataset):
    def __init__(self,vec,roi,surf,transform=None):
        #vec=np.reshape(vec,(len(ds),-1,ds.shape[1]))
        self.vec=vec #8000*14*3
        self.roi=roi
        self.transform = transform
        self.surf = surf

    def __getitem__(self, index: int):
        index1=index
        index2=np.random.randint(0,len(self.vec))
        fiber1=self.vec[index1]
        fiber2 = self.vec[index2]
        # fiber_norm1=self.x_norm[index1]
        # fiber_norm2 = self.x_norm[index2]
        img1= _feat_to_3D(np.expand_dims(fiber1,0), repeat_time=14).squeeze()
        img1=img1.transpose(2,0,1)
        img2= _feat_to_3D(np.expand_dims(fiber2,0), repeat_time=14).squeeze()
        img2=img2.transpose(2,0,1)
        roi1=self.roi[index1]
        surf1 = self.surf[index1]
        #roi2 = self.roi[index2]
        similarity=fiber_pair_similarity(fiber1, fiber2)
        img1=torch.tensor(img1, dtype=torch.float)
        img2 = torch.tensor(img2, dtype=torch.float)
        similarity = torch.tensor(similarity, dtype=torch.float)
        # if self.transform is not None:
        #     img = self.transform(img)
        return img1,img2,similarity,roi1,surf1,index

    def __len__(self) -> int:
        return len(self.vec)

class FiberCom_pair(data.Dataset):
    def __init__(self,vec,roi,transform=None):
        #vec=np.reshape(vec,(len(ds),-1,ds.shape[1]))
        self.vec=vec #8000*14*3
        self.roi=roi
        self.transform = transform

    def __getitem__(self, index: int):
        index1=index
        index2=np.random.randint(0,len(self.vec))
        fiber1=self.vec[index1]
        fiber2 = self.vec[index2]
        # fiber_norm1=self.x_norm[index1]
        # fiber_norm2 = self.x_norm[index2]
        img1= _feat_to_3D(np.expand_dims(fiber1,0), repeat_time=14).squeeze()
        img1=img1.transpose(2,0,1)
        img2= _feat_to_3D(np.expand_dims(fiber2,0), repeat_time=14).squeeze()
        img2=img2.transpose(2,0,1)
        roi1=self.roi[index1]
        #roi2 = self.roi[index2]
        similarity=fiber_pair_similarity(fiber1, fiber2)
        img1=torch.tensor(img1, dtype=torch.float)
        img2 = torch.tensor(img2, dtype=torch.float)
        similarity = torch.tensor(similarity, dtype=torch.float)
        fiber1=torch.tensor(fiber1.T, dtype=torch.float)
        fiber2 = torch.tensor(fiber2.T, dtype=torch.float)
        # if self.transform is not None:
        #     img = self.transform(img)
        return img1,img2,fiber1,fiber2,similarity,roi1,index

    def __len__(self) -> int:
        return len(self.vec)
class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, small=False, full=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.full = full

        if full:
            self.train = True

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.train_data, self.train_labels = torch.load(os.path.join(self.root, self.processed_folder, self.training_file))
        self.test_data, self.test_labels = torch.load(os.path.join(self.root, self.processed_folder, self.test_file))

        if full:
            self.train_data = np.concatenate((self.train_data, self.test_data), axis=0)
            self.train_labels = np.concatenate((self.train_labels, self.test_labels), axis=0)

        if small:
            self.train_data = self.train_data[0:1400]
            self.train_labels = self.train_labels[0:1400]
            if not full:
                self.train_data = self.train_data[0:1200]
                self.train_labels = self.train_labels[0:1200]
            self.test_data = self.test_data[0:200]
            self.test_labels = self.test_labels[0:200]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.full:
            img = Image.fromarray(img, mode='L')
        else:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]


class EMNIST(MNIST):
    """`EMNIST <https://www.nist.gov/itl/iad/image-group/emnist-dataset/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
            which one to use.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
    splits = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')

    def __init__(self, root, split, **kwargs):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        self.split = split
        self.training_file = self._training_file(split)
        self.test_file = self._test_file(split)
        super(EMNIST, self).__init__(root, **kwargs)

    def _training_file(self, split):
        return 'training_{}.pt'.format(split)

    def _test_file(self, split):
        return 'test_{}.pt'.format(split)

    def download(self):
        """Download the EMNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip
        import shutil
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print('Downloading ' + self.url)
        data = urllib.request.urlopen(self.url)
        filename = self.url.rpartition('/')[2]
        raw_folder = os.path.join(self.root, self.raw_folder)
        file_path = os.path.join(raw_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print('Extracting zip archive')
        with zipfile.ZipFile(file_path) as zip_f:
            zip_f.extractall(raw_folder)
        os.unlink(file_path)
        gzip_folder = os.path.join(raw_folder, 'gzip')
        for gzip_file in os.listdir(gzip_folder):
            if gzip_file.endswith('.gz'):
                print('Extracting ' + gzip_file)
                with open(os.path.join(raw_folder, gzip_file.replace('.gz', '')), 'wb') as out_f, \
                        gzip.GzipFile(os.path.join(gzip_folder, gzip_file)) as zip_f:
                    out_f.write(zip_f.read())
        shutil.rmtree(gzip_folder)

        # process and save as torch files
        for split in self.splits:
            print('Processing ' + split)
            training_set = (
                read_image_file(os.path.join(raw_folder, 'emnist-{}-train-images-idx3-ubyte'.format(split))),
                read_label_file(os.path.join(raw_folder, 'emnist-{}-train-labels-idx1-ubyte'.format(split)))
            )
            test_set = (
                read_image_file(os.path.join(raw_folder, 'emnist-{}-test-images-idx3-ubyte'.format(split))),
                read_label_file(os.path.join(raw_folder, 'emnist-{}-test-labels-idx1-ubyte'.format(split)))
            )
            with open(os.path.join(self.root, self.processed_folder, self._training_file(split)), 'wb') as f:
                torch.save(training_set, f)
            with open(os.path.join(self.root, self.processed_folder, self._test_file(split)), 'wb') as f:
                torch.save(test_set, f)

        print('Done!')


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)