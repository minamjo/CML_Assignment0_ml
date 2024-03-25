from model import *
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

def generate_image(input_image_path, save_image_path, image_size, model_path='models/gen_pix_195.pth', device='cuda'):
    # Load your model
    model = Generator(in_channels=3).to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    # Load an image
    input_img = Image.open(input_image_path)
    transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    input_image = transform(input_img).unsqueeze(0)
    with torch.no_grad():
        output_image = model(input_image.to(device)).cpu() * 0.5 + 0.5
        save_image(output_image, save_image_path)

def rmse(img_gen, img_lab):
    img_lab = img_lab.astype('float32')
    img_gen = img_gen.astype('float32')

    return np.linalg.norm(img_lab - img_gen)/(255.0*(img_lab.size**0.5))


if __name__ == '__main__':
    # please generate at least 10 images.
    pics = [10, 50,  100, 250, 500, 750, 1000, 3000, 5000, 7000, 9999]
    rmse_err = list()
    for i in range(len(pics)):
        input_image_path = 'dataset/input/img/input_image_%d.png' % pics[i]
        save_image_path = 'results/mod_gen_%d.png' % pics[i]
        generate_image(input_image_path, save_image_path, image_size=64)

        lab_img = cv2.imread('dataset/label/img/label_image_%d.png'% pics[i])
        gen_img = cv2.imread('results/gen_%d.png'% pics[i])
        gen_img = cv2.resize(gen_img, (160,120))
        rmse_err.append(rmse(gen_img, lab_img))
    print(np.mean(rmse_err))
