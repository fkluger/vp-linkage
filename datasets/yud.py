import glob
import os
import numpy as np
import scipy.io
import imageio
import lsd


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class YUDVP:

    def __init__(self, data_dir_path, split='', keep_in_mem=True, normalize_coords=False,
                 return_images=False):
        self.data_dir = data_dir_path

        self.image_folders = glob.glob(os.path.join(self.data_dir, "P*/"))

        self.keep_in_mem = keep_in_mem
        self.normalize_coords = normalize_coords
        self.return_images = return_images

        if split is not None:
            if split == "train" or split == 'val':
                self.set_ids = list(range(0, 25))
            elif split == "test":
                self.set_ids = list(range(25, 102))
            elif split == "all":
                self.set_ids = list(range(0, 102))
            else:
                assert False, "invalid split"

        self.dataset = [None for _ in self.set_ids]

        camera_params = scipy.io.loadmat(os.path.join(self.data_dir, "cameraParameters.mat"))

        f = camera_params['focal'][0, 0]
        ps = camera_params['pixelSize'][0, 0]
        pp = camera_params['pp'][0, :]

        self.K = np.matrix([[f / ps, 0, pp[0]], [0, f / ps, pp[1]], [0, 0, 1]])
        self.S = np.matrix([[2.0 / 640, 0, -1], [0, 2.0 / 640, -0.75], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.S*self.K) if normalize_coords else np.linalg.inv(self.K)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):

        scene_idx = self.set_ids[key]

        datum = self.dataset[key]

        if datum is None:
            image_path = glob.glob(os.path.join(self.image_folders[scene_idx], "P*.jpg"))[0]
            image_rgb = imageio.imread(image_path)
            image = rgb2gray(image_rgb)

            lsd_line_segments = lsd.detect_line_segments(image)

            if self.normalize_coords:
                lsd_line_segments[:, 0] -= 320
                lsd_line_segments[:, 2] -= 320
                lsd_line_segments[:, 1] -= 240
                lsd_line_segments[:, 3] -= 240
                lsd_line_segments[:, 0:4] /= 320.

            line_segments = np.zeros((lsd_line_segments.shape[0], 7+2+3+3))
            for li in range(line_segments.shape[0]):
                p1 = np.array([lsd_line_segments[li, 0], lsd_line_segments[li, 1], 1])
                p2 = np.array([lsd_line_segments[li, 2], lsd_line_segments[li, 3], 1])
                centroid = 0.5*(p1+p2)
                line = np.cross(p1, p2)
                line /= np.linalg.norm(line[0:2])
                line_segments[li, 0:3] = p1
                line_segments[li, 3:6] = p2
                line_segments[li, 6:9] = line
                line_segments[li, 9:12] = centroid
                line_segments[li, 12:15] = lsd_line_segments[li, 4:7]

            mat_gt_path = glob.glob(os.path.join(self.image_folders[scene_idx], "P*GroundTruthVP_CamParams.mat"))[0]
            gt_data = scipy.io.loadmat(mat_gt_path)

            true_vds = np.matrix(gt_data['vp'])
            true_vds[1, :] *= -1

            true_vps = np.array((self.K * true_vds).T)
            for vi in range(true_vps.shape[0]):
                true_vps[vi] /= true_vps[vi, 2]

            true_horizon = np.cross(true_vps[0], true_vps[1])

            datum = {'line_segments': line_segments, 'VPs': true_vps, 'id': scene_idx, 'VDs': true_vds,
                     'horizon': true_horizon}

            if self.return_images:
                datum['image'] = np.array(image_rgb)

            for vi in range(datum['VPs'].shape[0]):
                if self.normalize_coords:
                    datum['VPs'][vi, :][0] -= 320
                    datum['VPs'][vi, :][1] -= 240
                    datum['VPs'][vi, :][0:2] /= 320.
                datum['VPs'][vi, :] /= np.linalg.norm(datum['VPs'][vi, :])

            if self.keep_in_mem:
                self.dataset[key] = datum

        return datum


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = YUDVP("/tnt/data/scene_understanding/YUD",
                    split='train', normalize_coords=False, return_images=True)

    max_num_vp = 0
    max_num_ls = 0
    all_distances_smallest = []
    all_distances_second = []
    for idx in range(len(dataset)):
        vps = dataset[idx]['VPs']
        num_vps = vps.shape[0]
        if num_vps > max_num_vp: max_num_vp = num_vps
        num_ls = dataset[idx]['line_segments'].shape[0]

        if num_ls > max_num_ls: max_num_ls = num_ls

        ls = dataset[idx]['line_segments']

        distances_per_img = []

        plt.figure()
        plt.imshow(dataset[idx]['image'], cmap='gray')

        for li in range(ls.shape[0]):
            plt.plot([ls[li,0], ls[li,3]], [ls[li,1], ls[li,4]], '-', c='r')

        plt.show()
