
URLS = {
    "tokyo_xs": "https://drive.google.com/file/d/1kHl0xPVWJTNIdJXLOrm3xUS-SL8MvGKO/view?usp=sharing",
    "sf_xs": "https://drive.google.com/file/d/1OLRHvJqcgruNuVwd_ZVdX7RGvT8DqQQF/view?usp=sharing",
    "gsv_xs": "https://drive.google.com/file/d/1dnOQhDcT82o_jWLlcO8lS6kaeVdFMzvr/view?usp=sharing"


}
#   "tokyo_xs": "https://drive.google.com/file/d/1nf1RAas99gNNvoWEJOD8A4knpSsOTr-D/view?usp=sharing",
#     "sf_xs": "https://drive.google.com/file/d/1Rs9cPvGMs5RTuP7uPOaydE52c2dDeOxD/view?usp=sharing",
#     "gsv_xs": "https://drive.google.com/file/d/1RL1Vj9-HgAVWHcSW014OIxpFVJNpkUQr/view?usp=sharing"

# "tokyo_xs": "https://drive.google.com/file/d/15QB3VNKj93027UAQWv7pzFQO1JDCdZj2/view?usp=share_link",
#     "sf_xs": "https://drive.google.com/file/d/1tQqEyt3go3vMh4fj_LZrRcahoTbzzH-y/view?usp=share_link",
#     "gsv_xs": "https://drive.google.com/file/d/1q7usSe9_5xV5zTfN-1In4DlmF5ReyU_A/view?usp=share_link"

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

