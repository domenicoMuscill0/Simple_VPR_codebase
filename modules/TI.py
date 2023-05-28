import random
from glob import glob
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from typing import Tuple


class TemplateInjector(nn.Module):
    def __init__(self, image_dim: int, template_dim_reduction: Tuple[int, int] = (5, 4)):
        super(TemplateInjector, self).__init__()
        self.template_paths = glob(f"./templates/**/*.png", recursive=True)  # Path to templates to inject
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_dim = image_dim
        self.template_h = int(image_dim / template_dim_reduction[0])
        # Check that the template can fit into the image if we put it into the bottom 1/3 of the image height
        assert self.template_h < 1 / 3 * self.image_dim
        self.template_w = int(image_dim / template_dim_reduction[1])

        self.adjust_template = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize([self.template_h, self.template_w]),
            transforms.RandomRotation(45),
            transforms.ColorJitter(
                brightness=0.7,
                contrast=0.4,
                hue=0.2
            )
        ])

    def forward(self, x):
        n_images = x.shape[0]
        for i in range(n_images):
            template_path = random.choice(self.template_paths)
            template = Image.open(template_path).convert("RGB")
            template = self.adjust_template(template).to(self.device)
            # template = template.permute((1, 2, 0))
            tl_h, tl_w = random.randint(int(2 / 3 * self.image_dim), self.image_dim - self.template_h), \
                random.randint(self.template_w, self.image_dim - self.template_w)
            mask = (template != 0)
            x[i, :, tl_h:tl_h + self.template_h, tl_w:tl_w + self.template_w][mask] = \
                template[mask]

            del mask, template
        return x


if __name__ == '__main__':
    ti = TemplateInjector(224)
    x = ti.forward(torch.randn(224, 224, 3))
    tensorToImage = transforms.ToPILImage(mode="RGB")
    tensorToImage(x.permute((2, 0, 1))).save("../templates/test.png")
