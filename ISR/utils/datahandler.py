import os
import imageio
import numpy as np
from ISR.utils.logger import get_logger
from PIL import Image

class DataHandler:
    """
    DataHandler generate augmented batches used for training or validation.

    Args:
        lr_dir: directory containing the Low Res images.
        hr_dir: directory containing the High Res images.
        patch_size: integer, size of the patches extracted from LR images.
        scale: integer, upscaling factor.
        n_validation_samples: integer, size of the validation set. Only provided if the
            DataHandler is used to generate validation sets.
    """

    def __init__(self, lr_dir, hr_dir, patch_size, scale, n_validation_samples=None):
        self.folders = {'hr': hr_dir, 'lr': lr_dir}  # image folders
        self.extensions = ('.png', '.jpeg', '.jpg')  # admissible extension
        self.img_list = {}  # list of file names
        self.n_validation_samples = n_validation_samples
        self.patch_size = patch_size
        self.scale = scale
        self.patch_size = {'lr': patch_size, 'hr': patch_size * self.scale}
        self.logger = get_logger(__name__)
        print("make img list start")
        self._make_img_list()
        print("make img list finished")
        self._check_dataset()

    def _make_img_list(self):
        """ Creates a dictionary of lists of the acceptable images contained in lr_dir and hr_dir. """

        for res in ['hr', 'lr']:
            file_names = os.listdir(self.folders[res])
            file_names = [file for file in file_names if file.endswith(self.extensions)]
            

            # @shenghuiyu
            if res == 'hr':
                file_names=self._choose_from_sum(file_names, self.folders[res])

            self.img_list[res] = np.sort(file_names)

        if self.n_validation_samples:
            print(self.img_list)
            samples = np.random.choice(
                range(len(self.img_list['hr'])), self.n_validation_samples, replace=False
            )
            for res in ['hr', 'lr']:
                self.img_list[res] = self.img_list[res][samples]

    def _check_dataset(self):
        """ Sanity check for dataset. """

        # the order of these asserts is important for testing
        assert len(self.img_list['hr']) == self.img_list['hr'].shape[0], 'UnevenDatasets'
        assert self._matching_datasets(), 'Input/LabelsMismatch'

    def _matching_datasets(self):
        """ Rough file name matching between lr and hr directories. """
        # LR_name.png = HR_name_2.png
        # or
        # LR_name.png = HR_name_1.png
        LR_name_root = [x.split('.')[0].split('_')[1:3] for x in self.img_list['lr']]
        HR_name_root = [x.split('.')[0].split('_')[1:3] for x in self.img_list['hr']]
        
        return np.all(HR_name_root == LR_name_root)

    def _not_flat(self, patch, flatness):
        """
        Determines whether the patch is complex, or not-flat enough.
        Threshold set by flatness.
        """

        if max(np.std(patch, axis=0).mean(), np.std(patch, axis=1).mean()) < flatness:
            return False
        else:
            return True

    def _crop_imgs(self, imgs, batch_size, flatness):
        """
        Get random top left corners coordinates in LR space, multiply by scale to
        get HR coordinates.
        Gets batch_size + n possible coordinates.
        Accepts the batch only if the standard deviation of pixel intensities is above a given threshold, OR
        no patches can be further discarded (n have been discarded already).
        Square crops of size patch_size are taken from the selected
        top left corners.
        """

        slices = {}
        crops = {}
        crops['lr'] = []
        crops['hr'] = []
        accepted_slices = {}
        accepted_slices['lr'] = []
        top_left = {'x': {}, 'y': {}}
        n = 50 * batch_size
        for i, axis in enumerate(['x', 'y']):
            top_left[axis]['lr'] = np.random.randint(
                0, imgs['lr'].shape[i] - self.patch_size['lr'] + 1, batch_size + n
            )
            top_left[axis]['hr'] = top_left[axis]['lr'] * self.scale
        for res in ['lr', 'hr']:
            slices[res] = np.array(
                [
                    {'x': (x, x + self.patch_size[res]), 'y': (y, y + self.patch_size[res])}
                    for x, y in zip(top_left['x'][res], top_left['y'][res])
                ]
            )

        for slice_index, s in enumerate(slices['lr']):
            candidate_crop = imgs['lr'][s['x'][0] : s['x'][1], s['y'][0] : s['y'][1], slice(None)]
            if self._not_flat(candidate_crop, flatness) or n == 0:
                crops['lr'].append(candidate_crop)
                accepted_slices['lr'].append(slice_index)
            else:
                n -= 1
            if len(crops['lr']) == batch_size:
                break

        accepted_slices['hr'] = slices['hr'][accepted_slices['lr']]

        for s in accepted_slices['hr']:
            candidate_crop = imgs['hr'][s['x'][0] : s['x'][1], s['y'][0] : s['y'][1], slice(None)]
            crops['hr'].append(candidate_crop)

        crops['lr'] = np.array(crops['lr'])
        crops['hr'] = np.array(crops['hr'])
        return crops

    def _apply_transform(self, img, transform_selection):
        """ Rotates and flips input image according to transform_selection. """

        rotate = {
            0: lambda x: x,
            1: lambda x: np.rot90(x, k=1, axes=(1, 0)),  # rotate right
            2: lambda x: np.rot90(x, k=1, axes=(0, 1)),  # rotate left
        }

        flip = {
            0: lambda x: x,
            1: lambda x: np.flip(x, 0),  # flip along horizontal axis
            2: lambda x: np.flip(x, 1),  # flip along vertical axis
        }

        rot_direction = transform_selection[0]
        flip_axis = transform_selection[1]

        img = rotate[rot_direction](img)
        img = flip[flip_axis](img)

        return img

    def _transform_batch(self, batch, transforms):
        """ Transforms each individual image of the batch independently. """

        t_batch = np.array(
            [self._apply_transform(img, transforms[i]) for i, img in enumerate(batch)]
        )
        return t_batch

    def get_batch(self, batch_size, idx=None, flatness=0.0):
        """
        Returns a dictionary with keys ('lr', 'hr') containing training batches
        of Low Res and High Res image patches.

        Args:
            batch_size: integer.
            flatness: float in [0,1], is the patch "flatness" threshold.
                Determines what level of detail the patches need to meet. 0 means any patch is accepted.
        """

        if not idx:
            # randomly select one image. idx is given at validation time.
            idx = np.random.choice(range(len(self.img_list['hr'])))
        img = {}
        for res in ['lr', 'hr']:
            img_path = os.path.join(self.folders[res], self.img_list[res][idx])
            img[res] = imageio.imread(img_path) / 255.0
        batch = self._crop_imgs(img, batch_size, flatness)
        transforms = np.random.randint(0, 3, (batch_size, 2))
        batch['lr'] = self._transform_batch(batch['lr'], transforms)
        batch['hr'] = self._transform_batch(batch['hr'], transforms)

        return batch

    def get_validation_batches(self, batch_size):
        """ Returns a batch for each image in the validation set. """

        if self.n_validation_samples:
            batches = []
            for idx in range(self.n_validation_samples):
                batches.append(self.get_batch(batch_size, idx, flatness=0.0))
            return batches
        else:
            self.logger.error(
                'No validation set size specified. (not operating in a validation set?)'
            )
            raise ValueError(
                'No validation set size specified. (not operating in a validation set?)'
            )

    def get_validation_set(self, batch_size):
        """
        Returns a batch for each image in the validation set.
        Flattens and splits them to feed it to Keras's model.evaluate.
        """

        if self.n_validation_samples:
            batches = self.get_validation_batches(batch_size)
            valid_set = {'lr': [], 'hr': []}
            for batch in batches:
                for res in ('lr', 'hr'):
                    valid_set[res].extend(batch[res])
            for res in ('lr', 'hr'):
                valid_set[res] = np.array(valid_set[res])
            return valid_set
        else:
            self.logger.error(
                'No validation set size specified. (not operating in a validation set?)'
            )
            raise ValueError(
                'No validation set size specified. (not operating in a validation set?)'
            )

    # @shenghuiyu
    def _choose_from_sum(self, file_names, file_dir):
        list1 = []
        list2 = []
        output = []

        for f in file_names:
            if f[-5]=='1':
                list1.append(f)
            elif f[-5]=='2':
                list2.append(f)


        assert len(list1)==len(list2)


        for i in range(len(list1)):
            assert list1[i][:-5]==list2[i][:-5]

            img1=Image.open(file_dir+list1[i])
            img2=Image.open(file_dir+list2[i])
            # array = np.array(img)
            # print(len(array))

            sum1=0
            sum2=0
            for w in range(img1.size[0]):
                for h in range(img1.size[1]):
                    sum1+=sum(img1.getpixel((w,h)))
                    sum2+=sum(img2.getpixel((w,h)))

            if sum1 > sum2:
                output.append(list1[i])

            else:
                output.append(list2[i])

            print("getting image summation:"+str(i)+"/"+str(len(list1)))

        return output