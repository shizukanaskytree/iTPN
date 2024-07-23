import torch

conv = torch.nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2, padding=0)


x = torch.randn(2, 2, 6, 6)
print(x)
# print(x.reshape(2, -1, 3, 2, 3, 2).permute(0, 1, 3, 5, 2, 4).reshape(2, -1, 3, 3))
y = torch.nn.functional.unfold(input=x, kernel_size=3, stride=3, padding=0)
y = y.transpose(1, 2)
# print(y)
# print(y.shape)

z = torch.nn.functional.fold(y.transpose(1, 2), output_size=(6, 6), kernel_size=3, padding=0, stride=3)
print(z)

