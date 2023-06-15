


# URLS = {
#     "tokyo_xs": "https://drive.google.com/file/d/15QB3VNKj93027UAQWv7pzFQO1JDCdZj2/view?usp=share_link",
#     "sf_xs": "https://drive.google.com/file/d/1tQqEyt3go3vMh4fj_LZrRcahoTbzzH-y/view?usp=share_link",
#     "gsv_xs": "https://drive.google.com/file/d/1q7usSe9_5xV5zTfN-1In4DlmF5ReyU_A/view?usp=share_link"
# }

#QUESTI SONO DA GMAIL PRINCIPALE
# URLS = {
#     "tokyo_xs": "https://drive.google.com/file/d/1LpxOxwghR5Gfmx5qRCB3rbbWp9OannJi/view?usp=sharing",
#     "sf_xs": "https://drive.google.com/file/d/1dQt1iIKQAwUcUBkuBXCVkW4epMWbWR_i/view?usp=sharing",
#     "gsv_xs": "https://drive.google.com/file/d/16iaZMbuWOwmhGDUvNkEYdD4Ste-aFJVC/view?usp=sharing"
# }

#QUESTI SONO DA CFD-CWE
URLS = {
    "tokyo_xs": "https://drive.google.com/file/d/10DrpzZ7tl7OStgFDiTJasDbnGv_dxdh_/view?usp=sharing",
    "sf_xs": "https://drive.google.com/file/d/1usonhb5qdLsfCKX2zh1H-sfLoiBk_ioV/view?usp=sharing",
    "gsv_xs": "https://drive.google.com/file/d/1OSJOGt62aYimScMcoZl_De3TdJhi04OT/view?usp=sharing"
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

