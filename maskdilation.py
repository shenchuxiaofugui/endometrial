import shutil
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import os
from tqdm import tqdm
from RJSNRadiomicsFeatureExtractor import main_run



def batch_dilation(dirpath):
    # 一层一层外扩
    for case in tqdm(Path(dirpath).iterdir(), total=619):

        for roi_path in case.glob("*_roi_trans.nii.gz"):
            dilation_roi = sitk.ReadImage(str(roi_path))
            for i in range(1, 11):
                if os.path.exists(str(roi_path).replace("_roi_trans.nii.gz", f"_roi_dilation_{i}.nii.gz")):
                    continue
                try:
                    dilation_roi = sitk.DilateObjectMorphology(dilation_roi, (1, 1, 0), sitk.sitkBall)
                    erosion_roi = sitk.ErodeObjectMorphology(dilation_roi, (1, 1, 0), sitk.sitkBall)
                except (Exception, BaseException) as e:
                    print(case.name)
                    print(e)
                    continue
                sitk.WriteImage(dilation_roi,
                                str(roi_path).replace("_roi_trans.nii.gz", f"_roi_dilation_{i}.nii.gz"))
                sitk.WriteImage(erosion_roi,
                                str(roi_path).replace("_roi_trans.nii.gz", f"_roi_dilation_{i}.nii.gz"))



def direct_dilation(dirpath):
    # 直接外扩
    #flag = {"DWI":11, "T1CE":11, "T2":11}
    cases = [i for i in Path(dirpath).iterdir()]
    for case in tqdm(cases, total=len(cases)):
        if case.name != "zhaijunhua":
            continue
        os.makedirs(str(case) + "/direct_dilation", exist_ok=True)
        #os.makedirs(str(case) + "/direct_erosion", exist_ok=True)
        for modal in ["DWI", "T1CE", "T2"]:
            roi_path = str(case) + f"/{modal}_roi_trans.nii.gz"
            if not os.path.exists(roi_path):
                continue
            shutil.copyfile(str(case) + f"/{modal}_trans.nii.gz", str(case) + f"/direct_dilation/{modal}_trans.nii.gz")
            ori_roi = sitk.ReadImage(roi_path)
            for i in range(1, 11):
                if os.path.exists(str(case) + f"/direct_dilation/{modal}_roi_dilation_{i}.nii.gz"):
                #if os.path.exists(str(case) + f"/direct_erosion/{modal}_roi_erosion_{i}.nii.gz"):
                    continue
                try:
                    dilation_roi = sitk.DilateObjectMorphology(ori_roi, (i, i, 0), sitk.sitkBall)
                    # erosion_roi = sitk.ErodeObjectMorphology(ori_roi, (i, i, 0), sitk.sitkBall)
                    # erosion_roi = sitk.GrayscaleErode(ori_roi, kernelRadius=(i, i, 0), kernelType=sitk.sitkBall)
                except (Exception, BaseException) as e:
                    print(case.name, i)
                    print(e)
                    break
                sitk.WriteImage(dilation_roi, str(case) + f"/direct_dilation/{modal}_roi_dilation_{i}.nii.gz")
                dilation_array = sitk.GetArrayFromImage(dilation_roi)
                ori_array = sitk.GetArrayFromImage(ori_roi)
                new_array = dilation_array - ori_array
                new_roi = sitk.GetImageFromArray(new_array)
                sitk.WriteImage(new_roi,
                                str(case) + f"/direct_dilation/{modal}_dilation_{i}.nii.gz")
                # erosion_array = sitk.GetArrayFromImage(erosion_roi)
                # if np.sum(erosion_array) == 0:
                #     flag[modal] = i
                #     print(case.name, modal, i)
                #     break
                # dilation_roi = sitk.ReadImage(str(case) + f"/direct_dilation/{modal}_roi_dilation_{i}.nii.gz")
                # edge_array = dilation_array - erosion_array
                # edge_roi = sitk.GetImageFromArray(edge_array)
                # edge_roi.CopyInformation(ori_roi)
                # sitk.WriteImage(erosion_roi,
                #                 str(case) + f"/direct_erosion/{modal}_roi_erosion_{i}.nii.gz")
                # sitk.WriteImage(edge_roi,
                #                 str(case) + f"/direct_erosion/{modal}_edge_{i}.nii.gz")
    #print(flag)

def continue_dilation(dirpath):
    for case in tqdm(Path(dirpath).iterdir(), total=619):
        for modal in ["DWI", "T1CE", "T2"]:
            roi_path = str(case) + f"/{modal}_roi_dilation_10.nii.gz"
            dilation_roi = sitk.ReadImage(roi_path)
            for i in range(11, 17):
                dilation_roi = sitk.DilateObjectMorphology(dilation_roi, (1, 1, 0), sitk.sitkCross)
                sitk.WriteImage(dilation_roi, str(roi_path).replace("_roi_dilation_10.nii.gz", f"_roi_dilation_{i}.nii.gz"))

def extracted_features(modal, store_path):
    save_path = os.path.join(store_path, modal)
    os.makedirs(save_path, exist_ok=True)
    for i in tqdm([3, 5, 6,7,9]):
        save = os.path.join(save_path, f"{modal}_{i}_features.csv")
        main_run("/homes/syli/dataset/EC_all/outside/new_1.5T/clear_up_spacing", f"{modal}_trans", f"{modal}_dilation_{i}.nii.gz", save)

if __name__ == "__main__":
    dirpath = r"/homes/syli/dataset/EC_all/outside/shenzhen ROI seg/cases_spacing"
    direct_dilation(dirpath)
    extracted_features

