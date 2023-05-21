from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.transforms.functional import to_pil_image
import torch
import torchvision
import torchvision.transforms
from PIL import Image
import torchvision.transforms as T
import glob
import os

DIRECTORY = 'Dataset/Combined_V2/trainA'

weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)
model.eval()
model.to('cuda')

preprocess = weights.transforms()
preprocess.resize_size = [720]

print("starting")

i = 0
for filename in glob.glob('Dataset/Combined_V2/trainA/*.jpg'):
    img = read_image(filename)
    batch = preprocess(img).unsqueeze(0).to('cuda')

    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    mask = normalized_masks[0, class_to_idx["car"]]*2
    mask = torch.round(mask)

    img_org = Image.open(filename)
    img_org.putalpha(T.ToPILImage()(mask))
    # img_org = Image.fromarray(img * mask)
    pathname, extension = os.path.splitext(filename)
    name = pathname.split('/')[-1]
    img_org.save(f'Dataset/CarDataset_0.5/trainA/{name}.png')
    print(i)
    i += 1
print("ended")