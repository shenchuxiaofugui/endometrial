import shutil
from pathlib import Path
import tarfile
import os
import pandas as pd
import SimpleITK as sitk
import json

modal = "T2"
store_path = "/homes/syli/dataset/nnUNet/data/nnUNet_raw"
train_list = pd.read_csv(store_path+"/tv_index.csv")["ID"].values.tolist()
test_list = pd.read_csv(store_path+"/test_index.csv")["ID"].values.tolist()
def make_targz_one_by_one(output_filename, source_dir, i, split):
    img = sitk.ReadImage(source_dir+f"/{modal}.nii")
    sitk.WriteImage(img, output_filename+f"/images{split}/EC_{i}_0000.nii.gz")
    shutil.copyfile(source_dir+f"/{modal}_roi.nii.gz", output_filename+f"/labels{split}/EC_{i}.nii.gz")


store_path = store_path + f"/Dataset013_EC{modal}"
cases  = [i for i in Path("/homes/ydwang/Data/EC_diagnosis/EC_all_process_data").iterdir()]
os.makedirs(store_path+"/imagesTr", exist_ok=True)
os.makedirs(store_path+"/labelsTr", exist_ok=True)
os.makedirs(store_path+"/imagesTs", exist_ok=True)
os.makedirs(store_path+"/labelsTs", exist_ok=True)

data = []
for i, case in enumerate(cases):
    data.append(case.name)
    # if case.name in train_list:

    #     #make_targz_one_by_one(store_path, str(case), i, "Tr")
    #     continue
    # elif case.name in test_list:
    #     make_targz_one_by_one(store_path, str(case), i, "Ts")
    # else:
    #     print(case.name)
with open("mapping.json", "w") as file:
    json.dump(data, file)