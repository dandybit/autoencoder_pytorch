import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, img_size, embedded_dim=2):
        super().__init__()
        self.encoder = Encoder(img_size, embedded_dim)
        self.decoder = Decoder(self.encoder.output_before_flatten, embedded_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, img_size, embedded_dim):
        super().__init__()
        self.layer_1 = nn.Linear(embedded_dim, torch.prod(torch.tensor(img_size)))
        self.img_size = img_size
        self.layer_2 = nn.ConvTranspose2d(in_channels=self.img_size[-1], out_channels=128, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        self.layer_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.layer_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)

        self.layer_output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.layer_1(x)
        x = torch.nn.ReLU()(x)
        x = torch.reshape(x, (-1, self.img_size[-1], self.img_size[0], self.img_size[1]))
        x = self.layer_2(x)
        x = torch.nn.ReLU()(x)
        x = self.layer_3(x)
        x = torch.nn.ReLU()(x)
        x = self.layer_4(x)
        x = torch.nn.ReLU()(x)
        x = self.layer_output(x)
        x = torch.nn.Sigmoid()(x)

        return x


class Encoder(nn.Module):
    def __init__(self, img_size, embedded_dim):
        super().__init__()
        self.layer_1 = nn.Conv2d(in_channels=img_size[-1], out_channels=32, kernel_size=3, stride=2, padding=1)
        self.layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.layer_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.img_size = img_size
        self.output_before_flatten = (img_size[0] // 2 ** 3, img_size[1] // 2 ** 3, self.layer_3.out_channels)
        self.output_after_flatten = (img_size[0] // 2 ** 3) * (img_size[1] // 2 ** 3) * self.layer_3.out_channels

        self.layer_4 = nn.Linear(in_features=self.output_after_flatten, out_features=embedded_dim)

    def forward(self, x):
        x = self.layer_1(x)
        x = torch.nn.ReLU()(x)
        x = self.layer_2(x)
        x = torch.nn.ReLU()(x)
        x = self.layer_3(x)
        x = torch.nn.ReLU()(x)
        x = self.flatten(x)
        x = self.layer_4(x)

        return x


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),  # transform image to [0, 1] range
        transforms.Pad((2, 2, 2, 2), fill=0, padding_mode='constant')
        # transforms.Normalize()
    ])
    train_dataset = datasets.FashionMNIST(root='data', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root='data', train=False, transform=transform, download=True)

    data_train = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    data_test = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    img_shape = train_dataset[0][0].squeeze(0).unsqueeze(-1).shape
    autoencoder = Autoencoder(img_shape)

    optimizer = Adam(autoencoder.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    epochs = 3
    # training
    for epoch in range(epochs):
        for x in data_train:
            src = x[0]
            output = autoencoder(src)
            loss = loss_fn(output, src)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(loss.item())

    results = []
    targets = []
    for image in data_test:
        src = image[0]
        trg = image[1]
        output = autoencoder.encoder(src)
        results += output.tolist()
        targets += trg.tolist()

    results = torch.Tensor(results)
    targets = torch.Tensor(targets)

    colors = targets

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(results[:, 0], results[:, 1], c=colors, alpha=0.5, s=3)
    cbar = plt.colorbar(scatter)
    cbar.set_label('value')
    plt.show()

    # generate images
    max, min = np.max(results.numpy(), axis=(0, -1)), np.min(results.numpy(), axis=(0, -1))

    # new emb
    news_emb = torch.tensor(np.random.uniform(min, max, size=(16, 2)), dtype=torch.float32)

    images_gen = ((autoencoder.decoder(news_emb) * 255).squeeze(1).unsqueeze(-1)).detach().numpy()

    fig, axes = plt.subplots(4, 4, figsize=(10, 6))
    axes = axes.flatten()

    # display generated images
    for idx in range(images_gen.shape[0]):
        axes[idx].imshow(images_gen[idx], cmap='gray')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
