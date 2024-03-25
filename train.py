import os
import torch.utils.data as data
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
from model import *
class MyDataset(data.Dataset):
    def __init__(self, input_path, label_path, transform=None):
        self.input_path = [os.path.join(input_path, img_name) for img_name in os.listdir(input_path)]
        self.label_path = [os.path.join(label_path, img_name) for img_name in os.listdir(label_path)]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        img = Image.open(self.input_path[idx])
        label = Image.open(self.label_path[idx])
        if self.transform:
            img = self.transform(img)
            label = self.transform(label)
        return img, label

    def __len__(self):
        return len(self.label_path)


def test_examples(gen, test_dataloader, folder):
    inp, out = next(iter(test_dataloader))
    inp, out = inp.to(device), out.to(device)
    gen.eval()
    with torch.no_grad():
        fake = gen(inp)
        fake = fake * 0.5 + 0.5
        save_image(fake, folder + f"/gen.png")
        save_image(inp * 0.5 + 0.5, folder + "/input.png")
        save_image(out * 0.5 + 0.5, folder + "/label.png")
    gen.train()

def train(gen, dis, train_dl, OptimizerG, OptimizerD, L1_Loss, BCE_Loss, Gen_loss, Dis_loss, L1_LAMBDA):
    loop = tqdm(train_dl)
    for idx, (inp,out) in enumerate(loop):
        inp = inp.to(device)
        out = out.to(device)
        fake = gen(inp)
        D_real = dis(inp,out)
        D_real_loss = BCE_Loss(D_real, torch.ones_like(D_real))

        D_fake = dis(inp,fake.detach())
        D_fake_loss = BCE_Loss(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss)/2
        dis.zero_grad()
        D_loss.backward()
        OptimizerD.step()


        D_fake = dis(inp, fake)
        G_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
        L1 = L1_Loss(fake,out) * L1_LAMBDA
        G_loss = G_fake_loss + L1

        OptimizerG.zero_grad()
        G_loss.backward()
        OptimizerG.step()

        Gen_loss.append(G_loss.item())
        Dis_loss.append(D_loss.item())

def save_checkpoint(gen, dis, folder, epoch):
    torch.save(gen.state_dict, folder + 'gen_' + str(epoch) + '.pth')
    torch.save(dis.state_dict, folder + 'dis_' + str(epoch) + '.pth')
def test_train_dataloader(train_dataloader, batch_size):
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data

        # Check batch size
        assert inputs.size(0) == batch_size
        assert labels.size(0) == batch_size

        # Check data type
        assert isinstance(inputs, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

        if i < 3:
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title(f"Batch {i + 1}")
            plt.imshow(
                torchvision.utils.make_grid(inputs, padding=2, normalize=True).cpu().numpy().transpose((1, 2, 0)))
            plt.show()
        else:
            print("DataLoader works as expected!")
            return

def plot_loss(Gen_loss, Dis_loss):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(Gen_loss, label="Generator")
    plt.plot(Dis_loss, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    batch_size = 64
    image_size = 64
    nc = 3
    num_epochs = 121
    lr = 0.0002
    beta1 = 0.5
    ngpu = min(1, torch.cuda.device_count())
    input_dataroot = "dataset/input/img"
    label_dataroot = "dataset/label/img"
    my_data = MyDataset(input_dataroot, label_dataroot)
    l1_lambda = 100
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    Gen_loss = []
    Dis_loss = []

    train_size = int(0.8 * len(my_data))
    test_size = len(my_data) - train_size
    train_data, test_data = random_split(my_data, [train_size, test_size])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    dis = Discriminator(in_channels=nc).to(device)
    gen = Generator(in_channels=nc).to(device)
    OptimizerD = torch.optim.Adam(dis.parameters(), lr=lr, betas=(beta1, 0.999))
    OptimizerG = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))
    BCE_Loss = nn.BCEWithLogitsLoss()
    L1_Loss = nn.L1Loss()
    G_Scaler = torch.cuda.amp.GradScaler()
    D_Scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        train(
            gen, dis, train_dataloader, OptimizerG, OptimizerD, L1_Loss, BCE_Loss, G_Scaler, D_Scaler, Gen_loss, Dis_loss, l1_lambda
        )

        if epoch % 5 == 0:
            torch.save(gen.state_dict(), 'check/gen_%d.pth' % epoch)
            torch.save(gen.state_dict(), 'check/dis_%d.pth' % epoch)
    test_examples(gen, test_dataloader, 'ex')
    torch.save(gen.state_dict(), 'models/gen_64_120.pth')
    plot_loss(Gen_loss, Dis_loss)