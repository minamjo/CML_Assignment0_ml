## Model
Created a pix2pix network to perform paired image to image translation. The Generator model has similar architecture to UNET architecture. The discriminator is a patch-wise discriminator.

## Data
I created a mapping of the input and label images in the train.py in the MyDataset class. These (160, 120) images were resized to (64, 64).

## Training
Used both BCE_loss and L1_loss function and Adam optimizer. Training underwent for 121 epochs for a batch size of 64. After training, tested the model with the test data loader. Training and testing split was 0.8 and 0.2. The input, label, and test data batch images are present in the 'ex' folder.

## Evaluation
Evaluated the error of the generated images and the labelled images by calculating the rmse. Generated images using input images thatare saved to 'results' folder.

## References
pix2pix research paper: https://arxiv.org/pdf/1611.07004.pdf
pix2pix tutorial: https://medium.com/@Skpd/pix2pix-gan-for-generating-map-given-satellite-images-using-pytorch-6e50c318673a

