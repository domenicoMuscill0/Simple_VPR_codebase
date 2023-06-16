import os
import random
from glob import glob
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

# Load the pre-trained model
model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
model.eval()

# Define the image transforms
transform_images = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_counters = [
    ('vehicles', 0),
    ('people', 0),
    ('co', 0)
]

IMAGES_PER_CITY = 30


def is_blurry(image, threshold=1000):
    image = cv2.resize(image, (64, 64))
    score = cv2.Laplacian(image, cv2.CV_64F).var()
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    return abs(score) < threshold


def save_templates(input_image: Image, output_path: str = "../templates"):
    full_image = np.transpose(np.array(input_image), (2, 0, 1))

    # Make a prediction with the model
    with torch.no_grad():
        output = model(transform_images(input_image).unsqueeze(0))['out'][0]
        output_predictions = output.softmax(dim=0).argmax(0)

    # Select only the masks of vehicles, people, and common urban objects
    vehicles_mask = ((0 < output_predictions) * (output_predictions <= 7)).byte().cpu().numpy().astype(np.uint8)
    common_objects_mask = ((7 < output_predictions) * (output_predictions <= 14)).byte().cpu().numpy().astype(np.uint8)
    people_mask = (output_predictions >= 15).byte().cpu().numpy().astype(np.uint8)
    masks = [vehicles_mask, people_mask, common_objects_mask]

    for i, mask in enumerate(masks):
        # Save the raw segmentation data as a png image with a transparent background
        image = np.repeat(mask[np.newaxis, :, :], 3, axis=0) * full_image

        # Discard images with less than 10 pixels recognized
        if len(np.where(image != 0)[0]) <= 10:
            continue

        top, bottom, left, right = np.where(image != 0)[1].min(), \
            np.where(image != 0)[1].max(), \
            np.where(image != 0)[2].min(), \
            np.where(image != 0)[2].max()
        image = image[:, top:bottom, left:right]
        image = np.transpose(image, (1, 2, 0))

        # Discard too blurry images using image energy
        if is_blurry(image):
            continue

        image = Image.fromarray(image, mode='RGB')
        image = image.resize((224, 224), resample=Image.BICUBIC)
        obj_type, obj_counter = img_counters[i]
        image.save(output_path + f'/{obj_type}/{obj_counter}.png')
        img_counters[i] = (img_counters[i][0], img_counters[i][1] + 1)


if __name__ == '__main__':
    train_path = "../data/gsv_xs/train"
    for city in os.listdir(train_path):
        city_path = os.path.join(train_path, city)
        if not os.path.isdir(city_path):
            continue
        # Save at most 10 images for each city in gsv_xc/train
        for img_path in random.sample(glob(city_path + "/*.jpg"), IMAGES_PER_CITY):
            save_templates(Image.open(img_path).convert('RGB'))
