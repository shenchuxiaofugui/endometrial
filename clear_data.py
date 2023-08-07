import pandas as pd
import os
from pathlib import Path
import shutil
import re

root = Path("/homes/syli/dataset/EC_all/outside/new_1.5T/1.5new1")
for case in root.iterdir():
    if case.is_dir():
        new_name = re.findall(r"\d+", case.name)
        new_folder = f"/homes/syli/dataset/EC_all/outside/new_1.5T/clear_up/{new_name[0]}"
        files = [i for i in case.glob("*SEG.nii")]
        if len(files) != 3:
            print(case.name)
        else:
            os.makedirs(new_folder, exist_ok=True)
            for file in files:
                if "DWI" in str(file):
                    shutil.copyfile(str(file), new_folder+f"/DWI_roi.nii")
                    shutil.copyfile(str(file).replace("SEG", "MAIN"), new_folder+f"/DWI.nii")
                elif "T1WI" in str(file):
                    shutil.copyfile(str(file), new_folder+f"/T1CE_roi.nii")
                    shutil.copyfile(str(file).replace("SEG", "MAIN"), new_folder+f"/T1CE.nii")
                elif "T2WI" in str(file):
                    shutil.copyfile(str(file), new_folder+f"/T2_roi.nii")
                    try:
                        shutil.copyfile(str(file).replace("SEG", "MAIN"), new_folder+f"/T2.nii")
                    except:
                        shutil.copyfile(str(file).replace("SEG.nii", "MAIN.nii.gz"), new_folder+f"/T2.nii.gz")
                else:
                    print(case.name)






# df = pd.read_csv(root+f"/T1_features.csv")
# new_features = [i.replace("sst", "T1") for i in list(df)]
# df.columns = new_features
# df.to_csv(root+f"/T1_features.csv", index=False)


