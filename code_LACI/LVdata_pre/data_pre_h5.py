import random
import numpy as np
from tqdm import tqdm
import h5py
import nibabel as nib
import os
import scipy.ndimage


# output_size = [280, 280, 200]
output_size = [112, 112, 80]
def covert_h5():
    image_CT_path = '/data/chenjinfeng/Data/CT_160/image_CT'
    label_CT_path = '/data/chenjinfeng/Data/CT_160/label_CT'
    norm_h5_path = '/data/chenjinfeng/Data/CT_160/norm_112_80_h5'
    if not os.path.exists(norm_h5_path):
        os.makedirs(norm_h5_path)
    # List the files in the directories
    image_CT_files = os.listdir(image_CT_path) if os.path.exists(image_CT_path) else []
    label_CT_files = os.listdir(label_CT_path) if os.path.exists(label_CT_path) else []

    for item in tqdm(image_CT_files):
        image_data = nib.load(os.path.join(image_CT_path, item)).get_fdata()
        label_data = nib.load(os.path.join(label_CT_path, item)).get_fdata()

        # 先缩放图像和标签到1/2
        zoom_factors = [0.5, 0.5, 0.5]  # 缩放因子
        image_data = scipy.ndimage.zoom(image_data, zoom_factors, order=1)  # 使用线性插值
        label_data = scipy.ndimage.zoom(label_data, zoom_factors, order=0)  # 使用最近邻插值
        # 确定新的图像尺寸
        w, h, d = image_data.shape

        tempL = np.nonzero(label_data)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2

        # 确保边界计算考虑原始尺寸
        minx = max(minx - 25 - px, 0)
        maxx = min(maxx + 25 + px, w)
        miny = max(miny - 25 - py, 0)
        maxy = min(maxy + 25 + py, h)
        minz = max(minz - 25 - pz, 0)
        maxz = min(maxz + 25 + pz, d)

        # # 确保边界计算考虑原始尺寸
        # minx = max(minx - np.random.randint(10, 20) - px, 0)
        # maxx = min(maxx + np.random.randint(10, 20) + px, w)
        # miny = max(miny - np.random.randint(10, 20) - py, 0)
        # maxy = min(maxy + np.random.randint(10, 20) + py, h)
        # minz = max(minz - np.random.randint(5, 10) - pz, 0)
        # maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        image = (image_data[minx:maxx, miny:maxy, minz:maxz] - np.mean(image_data)) / np.std(image_data)
        image = image.astype(np.float32)
        label = label_data[minx:maxx, miny:maxy, minz:maxz]

        # 处理文件名和保存
        h5_filename = os.path.join(norm_h5_path, item.replace('.nii.gz', '_norm.h5'))
        f = h5py.File(h5_filename, 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()

def h5_to_nii(h5_directory, output_directory):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 列出所有H5文件
    h5_files = [f for f in os.listdir(h5_directory) if f.endswith('.h5')]

    for h5_file in tqdm(h5_files):
        h5_path = os.path.join(h5_directory, h5_file)

        # 读取H5文件
        with h5py.File(h5_path, 'r') as file:
            image = file['image'][:]
            label = file['label'][:]

        # 创建NIfTI图像
        image_nii = nib.Nifti1Image(image, affine=np.eye(4))
        label_nii = nib.Nifti1Image(label, affine=np.eye(4))

        # 输出文件路径
        image_output_path = os.path.join(output_directory, h5_file.replace('_norm.h5', '_image.nii.gz'))
        label_output_path = os.path.join(output_directory, h5_file.replace('_norm.h5', '_label.nii.gz'))

        # 保存为.nii.gz
        nib.save(image_nii, image_output_path)
        nib.save(label_nii, label_output_path)

# # 使用示例
covert_h5()
#
# h5_directory = '/data/chenjinfeng/Data/CT_160/norm_256_128_h5'
# output_directory = '/data/chenjinfeng/Data/CT_160/h5_2_nii'
# h5_to_nii(h5_directory, output_directory)


