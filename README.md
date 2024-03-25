## Model
Created a pix2pix network to perform paired image to image translation. The Generator model has similar architecture to UNET architecture. The discriminator is a patch-wise discriminator.

## Data
I created a mapping of the input and label images in the train.py in the MyDataset class. These (160, 120) images were resized to (64, 64).

## Training
Used both BCE_loss and L1_loss function and Adam optimizer. Training underwent for 121 epochs for a batch size of 64.

## Evaluation
Evaluated the error of the generated images and the labelled images by calculating the rmse.
