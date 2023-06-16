
URLS = {
    "tokyo_xs": "https://drive.google.com/file/d/1bY-2wV4HmfUQC_O_CWlDHYL_bOOifTMZ/view?usp=share_link",
    "sf_xs": "https://drive.google.com/file/d/1sNtAgMYjHZYRHyAdBB4XuyEKB7G452OJ/view?usp=share_link",
    "gsv_xs": "https://drive.google.com/file/d/1bJC4jWuFNio397PfVrQO53ET3zVvI08u/view?usp=share_link"
}


import os
import gdown
import shutil

os.makedirs("data", exist_ok=True)
for dataset_name, url in URLS.items():
    print(f"Downloading {dataset_name}")
    zip_filepath = f"data/{dataset_name}.zip"
    gdown.download(url, zip_filepath, fuzzy=True)
    shutil.unpack_archive(zip_filepath, extract_dir="data")
    os.remove(zip_filepath)

