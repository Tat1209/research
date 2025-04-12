from PIL import Image
from PIL import ImageDraw
from torchvision import transforms
from torchvision.transforms import InterpolationMode


img_path = "/root/app/aurora/competition01_gray_128x128/test/c01_20070404213027.jpg"

img = Image.open(img_path)

def blacken_region(x1, y1, x2, y2):
    def transform(image):
        draw = ImageDraw.Draw(image)
        draw.rectangle([x1, y1, x2, y2], fill=0)
        return image
    return transform

transform1 = transforms.Compose([
    transforms.CenterCrop(85),
    transforms.Lambda(blacken_region(0, 0, 24, 5)),
    transforms.Lambda(blacken_region(85-24, 0, 85-1, 5)),
])

transform = transforms.Compose([
    *transform1.transforms,
    # transforms.RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BILINEAR),
    # transforms.RandomRotation(degrees=(10, 10), interpolation=InterpolationMode.NEAREST),
    # transforms.RandomRotation(degrees=(10, 10), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomRotation(degrees=(10, 10), interpolation=InterpolationMode.BICUBIC),
    # transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ToTensor(), 
])

img = transform(img)

print(img.shape)

img = transforms.ToPILImage()(img)
img.save('test.jpg', quality=100, subsampling=0)


