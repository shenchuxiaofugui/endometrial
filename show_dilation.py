import numpy as np
import SimpleITK as sitk
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


dirpath = r"/homes/syli/dataset/EC_all/EC_114_spacing"
savepath = r'/homes/syli/dataset/EC_all/EC_check_dilation'


def show_img_label(img_array, roi_array, show_index, title):
    show_img = img_array[show_index, ...]
    show_roi = roi_array[show_index, ...]
    plt.title(title)
    plt.axis('off')
    plt.imshow(show_img, cmap='gray')
    show_roi = show_roi+0.1
    plt.contour(show_roi, linewidths=1, colors=["red", "yellow", "blue"])
    # show_roi = np.ma.masked_equal(show_roi, 0)
    # plt.imshow(show_roi, cmap="autumn", alpha=0.5)
    #plt.show()


def show_all_label(img_array, roi_array, savepath='', case='', modal=''):
    roi_index = [i for i, e in enumerate(np.sum(roi_array, axis=(1, 2)).tolist()) if e != 0]
    length = len(roi_index)
    if length > 6:
        roi_index = roi_index[int(length/2)-3:int(length/2)+3]

    #f = plt.subplots(figsize=(40, 40))
    # roi_max_index = np.argmax(np.sum(roi_array, axis=(1, 2)))
    for k in range(len(roi_index)):
        plt.subplot(2, 3, k+1)
        title = case + '_' +str(roi_index[k])
        show_img_label(img_array, roi_array, roi_index[k], title)
    if savepath!='':
        plt.savefig(savepath + f'/{case}_{modal}.jpg', bbox_inches='tight', dpi=400)
    else:
        plt.show()
    plt.close()


def show_label(dirpath, storepath):
    colors = ["lawngreen", "gold", "deepskyblue", "m", "forestgreen"]
    camps = ["spring", "cool", "Wistia"]
    for case in Path(dirpath).iterdir():
        for modal in ["DWI", "T1CE", "T2"]:
            img = sitk.ReadImage(str(case) + f"/{modal}_resampled.nii")
            img_arr = sitk.GetArrayFromImage(img)
            ori_roi = sitk.ReadImage(str(case) + f"/{modal}_roi_resampled.nii")
            roi_array = sitk.GetArrayFromImage(ori_roi)
            roi_max_index = np.argmax(np.sum(roi_array, axis=(1, 2)))
            show_img_label(img_arr, roi_array, roi_max_index, case.name+ "_"+ modal)
            for i in range(4):
                roi = sitk.ReadImage(str(case) + f"/direct_dilation/{modal}_roi_dilation_{2*i+2}.nii")
                roi_arr = sitk.GetArrayFromImage(roi)
                show_roi = roi_arr[roi_max_index, ...]
                plt.contour(show_roi, linewidths=0.1, colors=colors[i])
                # show_roi = np.ma.masked_equal(show_roi, 0)
                # plt.imshow(show_roi, cmap=camps[i], alpha=0.5)
            plt.savefig(storepath+f"/{case.name}_{modal}.jpg", bbox_inches='tight', dpi=400)
            plt.close()


def show_one_case(img_path, label_path, show_max=False, store_path=''):
    filepath, fullname = os.path.split(img_path)
    fname = fullname[:-7]
    img = sitk.ReadImage(img_path)
    img_arr = np.flip(sitk.GetArrayFromImage(img), axis=1)
    ori_roi = sitk.ReadImage(label_path)
    roi_array = np.flip(sitk.GetArrayFromImage(ori_roi), axis=1)
    if show_max:
        roi_max_index = np.argmax(np.sum(roi_array, axis=(1, 2)))
        show_img_label(img_arr, roi_array, roi_max_index, fname)
        if store_path != '':
            plt.savefig(savepath + f'/{store_path}_{fname}.jpg', bbox_inches='tight', dpi=400)
        else:
            plt.show()
    else:
        show_all_label(img_arr, roi_array, store_path, fname)


predict_path = "/homes/syli/python/nnunet/nnUNet_trained_models/nnUNet/3d_fullres/Task011_glioma/nnUNetTrainerV2BraTSRegions__nnUNetPlansv2.1/cv_niftis_postprocessed"
label_path = "/homes/syli/python/nnunet/nnUNet_trained_models/nnUNet/3d_fullres/Task011_glioma/nnUNetTrainerV2BraTSRegions__nnUNetPlansv2.1/gt_niftis"
image_path = "/homes/syli/python/nnunet/nnUNet_raw_data/nnUNet_raw_data/Task011_glioma/imagesTr"
store_path = "/homes/syli/python/nnunet/nnUNet_raw_data/nnUNet_raw_data/Task011_glioma/wrong"
for i in ["p53", "PA249n", "p23", "PA238n", "PA218n", "PA222n", "PA206n"]:
    show_one_case(image_path+f"/{i}_0003.nii.gz", label_path+f"/{i}.nii.gz", store_path=store_path+"/prediction")
    show_one_case(image_path + f"/{i}_0003.nii.gz", predict_path + f"/{i}.nii.gz", store_path=store_path+"/label")