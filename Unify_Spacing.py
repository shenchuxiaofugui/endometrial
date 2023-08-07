from pathlib import Path
import nibabel as nib
# import torch
from tqdm import tqdm
import os, shutil
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from copy import deepcopy
import numpy as np
# from NewReample import Resampler
import SimpleITK as sitk
from monai.transforms import Resize
import matplotlib.pyplot as plt


def register (file):
    img = nib.load(file)
    qform = img.get_qform()
    img.set_qform(qform)
    sfrom = img.get_sform()
    img.set_sform(sfrom)
    nib.save(img,file)


def check_image(root, label_key, print_infos = True):
    infos = {'spacing':[], 'fixed_path':[], 'shape':[],'count':[]}
    for case in tqdm(Path(root).iterdir(), total = len(os.listdir(root)),desc='Checking Spaicng'):
        if case.is_dir():
            for i in case.glob(f"{label_key}*"):

                spacing = nib.load(i).header['pixdim'][1:4]
                #(spacing)
                shape = nib.load(i).header.get_data_shape()
                if tuple(spacing) in infos['spacing']:
                    infos['count'][infos['spacing'].index(tuple(spacing))] += 1
                else:
                    infos['spacing'].append(tuple(spacing))
                    infos['fixed_path'].append(str(i))
                    infos['shape'].append(tuple(shape))
                    infos['count'].append(int(1))

    
    if print_infos:
        length = len(infos['spacing'])
        print('=' * 80)
        if length < len(os.listdir(root))/10:
            counts = infos['count']
            count = max(counts)
            index = counts.index(count)
            spacing = infos['spacing'][index]
            path = infos['fixed_path'][index]           
            print(f'The most frequently occurring situation:'+
                    f'{spacing, path, count} in {length} situations')
        else:
            x,y,z = [], [], []
            for spacing in infos["spacing"]:
                if spacing[0] not in x: x.append(spacing[0])
                if spacing[1] not in y: y.append(spacing[1])
                if spacing[2] not in z: z.append(spacing[2])
            spacing = [np.percentile(values, 50) for values in [x,y,z]]
            path = None
            print(f'The medians of spacing in axes of x, y, z is {tuple(spacing)})')
        print('=' * 80)
        return infos, spacing, path
    
# root = '/homes/dxli/Data/esophageal_cancer_raw2/Images_to_extract/CT_error'
# dst = '/homes/dxli/Data/esophageal_cancer_raw2/Spacing_Unified_Image/CT_error'

def ResizeSipmleITKImage(image, expected_spacing=None, expected_shape=None, method=sitk.sitkNearestNeighbor, key = 'image'):
    '''
    Resize the SimpleITK image. One of the expected resolution/spacing and final shape should be given.

    :param image: The SimpleITK image.
    :param expected_resolution: The expected resolution.
    :param excepted_shape: The expected final shape.
    :return: The resized image.

    Apr-27-2018, Yang SONG [yang.song.91@foxmail.com]
    '''


    if (expected_spacing is None) and (expected_shape is None):
        print('Give at least one parameters. ')
        return image

    shape = image.GetSize()
    resolution = image.GetSpacing()

    if expected_spacing is None:
        dim_0, dim_1, dim_2 = False, False, False
        if expected_shape[0] < 1e-6:
            expected_shape[0] = shape[0]
            dim_0 = True
        if expected_shape[1] < 1e-6:
            expected_shape[1] = shape[1]
            dim_1 = True
        if expected_shape[2] < 1e-6:
            expected_shape[2] = shape[2]
            dim_2 = True
        expected_spacing = [raw_resolution * raw_size / dest_size for dest_size, raw_size, raw_resolution in
                               zip(expected_shape, shape, resolution)]
        if dim_0: expected_spacing[0] = resolution[0]
        if dim_1: expected_spacing[1] = resolution[1]
        if dim_2: expected_spacing[2] = resolution[2]
        
    elif expected_shape is None:
        dim_0, dim_1, dim_2 = False, False, False
        if expected_spacing[0] < 1e-6: 
            expected_spacing[0] = resolution[0]
            dim_0 = True
        if expected_spacing[1] < 1e-6: 
            expected_spacing[1] = resolution[1]
            dim_1 = True
        if expected_spacing[2] < 1e-6: 
            expected_spacing[2] = resolution[2]
            dim_2 = True
        expected_shape = [int(raw_resolution * raw_size / dest_resolution) for
                       dest_resolution, raw_size, raw_resolution in zip(expected_spacing, shape, resolution)]
        if dim_0: expected_shape[0] = shape[0]
        if dim_1: expected_shape[1] = shape[1]
        if dim_2: expected_shape[2] = shape[2]

    # output = sitk.Resample(image, expected_shape, sitk.AffineTransform(len(shape)), method, image.GetOrigin(),
    #                        expected_resolution, image.GetDirection(), dtype)
    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(expected_shape)		# 目标图像大小
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputSpacing(np.array(expected_spacing ,dtype='double').tolist())
    # 根据需要重采样图像的情况设置不同的dype
    if key =='image':
        resampler.SetOutputPixelType(sitk.sitkFloat32)   # 线性插值用于PET/CT/MRI之类的，保存float32
    elif key == 'label' or key == 'mask':
        resampler.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))    
    resampler.SetInterpolator(method)
    output = resampler.Execute(image)  # 得到重新采样后的图像
    return output

def ResizeNiiFile(file_path, store_path, expected_resolution=None, expected_shape=None, key = 'image'):
    expected_resolution = deepcopy(expected_resolution)
    expected_shape = deepcopy(expected_shape)
    image = sitk.ReadImage(str(file_path))
    if 'image' in str(file_path):
        method = sitk.sitkBSpline
    elif 'label' in str(file_path):
        method = sitk.sitkNearestNeighbor
    else:
        method = sitk.sitkNearestNeighbor
    resized_image = ResizeSipmleITKImage(image, expected_resolution, expected_shape, method=method, key=key)
    sitk.WriteImage(resized_image, str(store_path))


def unify_spacing(root, dst_folder, modal_list, spacings=None, image_key = 'image'):

    # resamper = Resampler()
    for case in tqdm(Path(root).iterdir(), total=len(os.listdir(root)), desc='Processing'):
        if case.is_dir():
            if not (Path(dst_folder)/case.name).exists():
                Path.mkdir(Path(dst_folder)/case.name, exist_ok=True)
            for i in modal_list:
                # if 'roi' in str(i):
                #     continue
                # try:
                spacing = spacings[i]
                filename = os.path.join(str(case), i + '.nii')
                roiname = filename.replace('.nii', '_roi.nii.gz')
                try:
                    input_nii = sitk.ReadImage(filename)
                    roi = sitk.ReadImage(roiname)
                except:
                    print(case.name)
                    break

                # except:
                #     register(str(i))
                #     image = sitk.ReadImage(str(i))
                #     print(i)
                img_store = Path(dst_folder)/case.name/ (i + '_resampled.nii')
                roi_store = Path(dst_folder) / case.name / (i + '_roi_resampled.nii.gz')
                if input_nii.GetSpacing() == spacing:
                    shutil.copy(filename, img_store)
                    shutil.copy(roiname, roi_store)
                    continue
                else:
                    resized_image = ResizeSipmleITKImage(input_nii,  expected_spacing=spacing, method=sitk.sitkBSpline)
                    sitk.WriteImage(resized_image, str(img_store))
                    # roi_Reg = resamper.ResampleToReference(roi, resized_image, sitk.sitkLinear,
                    #                                        save_path=str(roi_store),
                    #                                        is_roi=True)
                    resized_roi = ResizeSipmleITKImage(roi, expected_spacing=spacing, method=sitk.sitkNearestNeighbor, key="label")
                    sitk.WriteImage(resized_roi, str(roi_store))

                    roi_Reg_arr = sitk.GetArrayFromImage(resized_roi)
                    if np.sum(roi_Reg_arr) == 0:
                        print("wrong", case.name)



def re_resize():
    for case in Path(dst).iterdir():
        for modal in ['DWI', 'T1CE', 'T2']:
            spacing = spacings[modal]
            modal += '_roi_resampled.nii.gz'
            roi_path = str(case)+f'/{modal}'
            roi = sitk.ReadImage(roi_path)
            roi_arr = sitk.GetArrayFromImage(roi)
            if np.sum(roi_arr) == 0:
                print(case.name, modal)
                ori_roi_path = roi_path.replace('spacing', 'all_process_data').replace('_resampled', '')
                ori_roi = sitk.ReadImage(ori_roi_path)
                ori_roi_arr = sitk.GetArrayFromImage(ori_roi)
                if np.sum(ori_roi_arr) != 0:
                    print("人家明明有的")
                print(roi_arr.shape)
                resize = Resize(spatial_size=roi_arr.shape, mode='trilinear')
                ori_roi_arr = ori_roi_arr.astype(np.float32)[np.newaxis,:,:,:]
                #new_roi = ResizeSipmleITKImage(ori_roi, expected_spacing=spacing, method=sitk.sitkNearestNeighbor)
                # ori_roi_arr = torch.Tensor(ori_roi_arr)
                # print(ori_roi_arr.dim())
                new_roi_arr = resize(ori_roi_arr)
                new_roi_arr = np.squeeze(new_roi_arr)
                new_roi = sitk.GetImageFromArray(new_roi_arr)
                new_roi.CopyInformation(roi)
                if np.sum(new_roi_arr) == 0:
                    print(case.name, modal, "哈哈哈哈它没救了")
                else:
                    sitk.WriteImage(new_roi, roi_path)

def show_roi(roi_array, k, img_arr = None, title = None, roi_index = None):
    plt.subplot(2, 2, k)
    plt.title(title)
    if roi_index:
        roi_max_index = roi_index
    else:
        roi_max_index = np.argmax(np.sum(roi_array, axis=(1, 2)))
    show_roi = roi_array[roi_max_index, ...]
    mask = np.ma.masked_equal(show_roi, 0)
    if img_arr.any():
        show_img = img_arr[roi_max_index, ...]
        plt.imshow(show_img, cmap='gray')
    plt.imshow(mask, 'autumn')
    plt.axis('off')


def monai_resize(roi_arr, shape):
    resize = Resize(spatial_size=shape, mode='trilinear')
    ori_roi_arr = roi_arr.astype(np.float32)[np.newaxis, :, :, :]
    new_roi_arr = resize(ori_roi_arr)
    new_roi_arr = np.squeeze(new_roi_arr)
    new_roi_arr = np.round(new_roi_arr)
    print(new_roi_arr.min(), new_roi_arr.max())
    return new_roi_arr

def compare(casedir, modals, spacings):
    resampler = Resampler()
    for case in Path(casedir).iterdir():
        for modal in modals:
            spacing = spacings[modal]
            img = sitk.ReadImage(str(case / f'{modal}.nii'))
            img_arr = sitk.GetArrayFromImage(img)
            roi = sitk.ReadImage(str(case / f'{modal}_roi.nii'))
            roi_arr = sitk.GetArrayFromImage(roi)
            resized_img = ResizeSipmleITKImage(img, expected_spacing=spacing, method=sitk.sitkBSpline)
            resized_img_arr = sitk.GetArrayFromImage(resized_img)
            zyh_roi = resampler.ResampleToReference(roi, resized_img, sitk.sitkLinear, is_roi=True)
            zyh_roi_arr = sitk.GetArrayFromImage(zyh_roi)
            near_roi = ResizeSipmleITKImage(roi,  expected_spacing=spacing, method=sitk.sitkNearestNeighbor)
            near_roi_arr = sitk.GetArrayFromImage(near_roi)
            monai_roi = monai_resize(roi_arr, resized_img_arr.shape)
            show_roi(roi_arr, 1, img_arr,'original')
            show_roi(zyh_roi_arr, 2, resized_img_arr, 'linear')
            show_roi(near_roi_arr, 3, resized_img_arr, 'nearest')
            show_roi(monai_roi, 4, resized_img_arr, 'trilinear')
            plt.show()


def compare_double(casedir, modals):
    resampler = Resampler()
    for case in Path(casedir).iterdir():
        for modal in modals:
            img = sitk.ReadImage(str(case / f'{modal}.nii'))
            img_arr = sitk.GetArrayFromImage(img)
            roi = sitk.ReadImage(str(case / f'{modal}_roi.nii'))
            roi_arr = sitk.GetArrayFromImage(roi)
            a = list(img.GetSize())
            a[2] = a[2] * 2
            resized_img = ResizeSipmleITKImage(img, expected_shape=a, method=sitk.sitkBSpline)
            resized_img_arr = sitk.GetArrayFromImage(resized_img)
            zyh_roi = resampler.ResampleToReference(roi, resized_img, sitk.sitkLinear, is_roi=True)
            zyh_roi_arr = sitk.GetArrayFromImage(zyh_roi)
            near_roi = ResizeSipmleITKImage(roi,  expected_shape=a, method=sitk.sitkNearestNeighbor)
            near_roi_arr = sitk.GetArrayFromImage(near_roi)
            monai_roi = monai_resize(roi_arr, resized_img_arr.shape)
            roi_max_index = np.argmax(np.sum(roi_arr, axis=(1, 2)))
            if np.sum(roi_arr[roi_max_index + 1,:,:]) > np.sum(roi_arr[roi_max_index - 1,:,:]):
                show_index = 2*roi_max_index+1
            else:
                show_index = 2*roi_max_index-1
            print(roi_arr.shape, zyh_roi_arr.shape, near_roi_arr.shape, monai_roi.shape)
            show_roi(roi_arr, 1, img_arr, 'original', roi_max_index)
            show_roi(zyh_roi_arr, 2, resized_img_arr, 'linear', show_index)
            show_roi(near_roi_arr, 3, resized_img_arr, 'nearest', show_index)
            show_roi(monai_roi, 4, resized_img_arr, 'trilinear', show_index)
            plt.title(f"{case.name}_{modal}")
            plt.show()


def copy_info_from_img(root):
    for case in Path(root).iterdir():
        for roi_path in case.glob("*_roi.nii.gz"):
            roi = sitk.ReadImage(str(roi_path))
            img = sitk.ReadImage(str(roi_path).replace("_roi", "").replace(".gz", ""))
            try:
                img.CopyInformation(roi)
            except:
                print(case.name)
            sitk.WriteImage(img, str(roi_path).replace("_roi", "").replace(".gz", ""))


def copy_direct_from_T2(root):
    for case in Path(root).iterdir():
        T1CE_path = str(case) + "/T1CE_resampled.nii"
        if os.path.exists(T1CE_path):
            roi_path = str(case) + "/T1CE_roi_resampled.nii.gz"
            img = sitk.ReadImage(T1CE_path)
            roi = sitk.ReadImage(roi_path)
            T2_path = str(case) + "/T2_resampled.nii"
            T2_img = sitk.ReadImage(T2_path)
            dirction = T2_img.GetDirection()
            img.SetDirection(dirction)
            roi.SetDirection(dirction)
            sitk.WriteImage(img, T1CE_path)
            sitk.WriteImage(roi, roi_path)




if __name__ == '__main__':
    # ResizeNiiFile()
    root = '/homes/syli/dataset/EC_all/outside/yfy/buchong'
    # root = '/homes/syli/dataset/EC_all/EC_all_process_data'
    dst = '/homes/syli/dataset/EC_all/outside/yfy/buchong_114'
    spacings = {'DWI': [1, 1, 4], 'T1CE': [1, 1, 4], 'T2': [1, 1, 4]}
    #check_image(root, "DWI_roi.nii")
    # unify_spacing(root, dst,['DWI', 'T1CE', 'T2'],
    #     #fixed_path="/homes/dxli/Data/esophageal_cancer_raw2/Spacing_Unified_Image/CT/664763/image_resampled.nii.gz",
    #     spacings = {'DWI':[1.875, 1.875, 5], 'T1CE':[0.96875, 0.96875, 5.2], 'T2':[1.05, 1.05, 5.2]}
    #
    # )
    # compare_double(root, ['DWI', 'T1CE', 'T2'])
    unify_spacing(root, dst, ["DWI", "T1CE", "T2"], spacings)
    # copy_info_from_img(root)
    # copy_direct_from_T2(root)





