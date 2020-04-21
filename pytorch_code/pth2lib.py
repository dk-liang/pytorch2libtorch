import torch
import torchvision.models as models
import torch.nn as nn
import os
from PIL import Image
from torchvision import datasets, transforms
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

path ='./model/model.pt'
w = 1280
h = 720

torch_model = torch.load(path).cuda() # load pytorch model
batch_size = 1  #batch size
input_shape = (3, h, w)   #git the input shape

example = torch.randn(batch_size, *input_shape).cuda()


traced_script_module = torch.jit.trace(torch_model, example).cuda()
img = Image.open('./test.jpg').convert('RGB')
img.resize((w,h))

transforms1 = transforms.ToTensor()
img = transforms1(img)

transforms2 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img = transforms2(img).unsqueeze(0)
print(img.shape)
output = traced_script_module(img.cuda())

print(output)
# ----------------------------------
traced_script_module.save('./model/model_transfer.pt')
print("model has been transfered")
