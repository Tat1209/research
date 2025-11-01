from functools import partial
from torchvision.transforms._presets import ImageClassification
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES


_COMMON_META = {
        "min_size": (1, 1),
        "categories": _IMAGENET_CATEGORIES,
        }


class ResNet18_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
            url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
            transforms=partial(ImageClassification, crop_size=224),
            meta={
                **_COMMON_META,
                "num_params": 11689512,
                "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
                "_metrics": {
                    "ImageNet-1K": {
                        "acc@1": 69.758,
                        "acc@5": 89.078,
                        }
                    },
                "_ops": 1.814,
                "_file_size": 44.661,
                "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
                },
            )
    DEFAULT = IMAGENET1K_V1


class ResNet34_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
            url="https://download.pytorch.org/models/resnet34-b627a593.pth",
            transforms=partial(ImageClassification, crop_size=224),
            meta={
                **_COMMON_META,
                "num_params": 21797672,
                "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
                "_metrics": {
                    "ImageNet-1K": {
                        "acc@1": 73.314,
                        "acc@5": 91.420,
                        }
                    },
                "_ops": 3.664,
                "_file_size": 83.275,
                "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
                },
            )
    DEFAULT = IMAGENET1K_V1


class ResNet50_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
            url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
            transforms=partial(ImageClassification, crop_size=224),
            meta={
                **_COMMON_META,
                "num_params": 25557032,
                "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
                "_metrics": {
                    "ImageNet-1K": {
                        "acc@1": 76.130,
                        "acc@5": 92.862,
                        }
                    },
                "_ops": 4.089,
                "_file_size": 97.781,
                "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
                },
            )
    DEFAULT = IMAGENET1K_V1


class ResNet101_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
            url="https://download.pytorch.org/models/resnet101-63fe2227.pth",
            transforms=partial(ImageClassification, crop_size=224),
            meta={
                **_COMMON_META,
                "num_params": 44549160,
                "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
                "_metrics": {
                    "ImageNet-1K": {
                        "acc@1": 77.374,
                        "acc@5": 93.546,
                        }
                    },
                "_ops": 7.801,
                "_file_size": 170.511,
                "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
                },
            )
    DEFAULT = IMAGENET1K_V1


class ResNet152_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
            url="https://download.pytorch.org/models/resnet152-394f9c45.pth",
            transforms=partial(ImageClassification, crop_size=224),
            meta={
                **_COMMON_META,
                "num_params": 60192808,
                "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
                "_metrics": {
                    "ImageNet-1K": {
                        "acc@1": 78.312,
                        "acc@5": 94.046,
                        }
                    },
                "_ops": 11.513,
                "_file_size": 230.434,
                "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
                },
            )
    DEFAULT = IMAGENET1K_V1


class ResNeXt50_32X4D_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
            url="https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
            transforms=partial(ImageClassification, crop_size=224),
            meta={
                **_COMMON_META,
                "num_params": 25028904,
                "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
                "_metrics": {
                    "ImageNet-1K": {
                        "acc@1": 77.618,
                        "acc@5": 93.698,
                        }
                    },
                "_ops": 4.23,
                "_file_size": 95.789,
                "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
                },
            )
    DEFAULT = IMAGENET1K_V1


class ResNeXt101_32X8D_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
            url="https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
            transforms=partial(ImageClassification, crop_size=224),
            meta={
                **_COMMON_META,
                "num_params": 88791336,
                "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
                "_metrics": {
                    "ImageNet-1K": {
                        "acc@1": 79.312,
                        "acc@5": 94.526,
                        }
                    },
                "_ops": 16.414,
                "_file_size": 339.586,
                "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
                },
            )
    DEFAULT = IMAGENET1K_V1


class ResNeXt101_64X4D_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
            url="https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
            transforms=partial(ImageClassification, crop_size=224),
            meta={
                **_COMMON_META,
                "num_params": 83455272,
                "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
                "_metrics": {
                    "ImageNet-1K": {
                        "acc@1": 78.956,
                        "acc@5": 94.252,
                        }
                    },
                "_ops": 15.46,
                "_file_size": 319.318,
                "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
                },
            )
    DEFAULT = IMAGENET1K_V1


class Wide_ResNet50_2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
            url="https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
            transforms=partial(ImageClassification, crop_size=224),
            meta={
                **_COMMON_META,
                "num_params": 68883240,
                "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#wide-resnet",
                "_metrics": {
                    "ImageNet-1K": {
                        "acc@1": 78.468,
                        "acc@5": 94.086,
                        }
                    },
                "_ops": 11.398,
                "_file_size": 263.124,
                "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
                },
            )
    DEFAULT = IMAGENET1K_V1


class Wide_ResNet101_2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
            url="https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
            transforms=partial(ImageClassification, crop_size=224),
            meta={
                **_COMMON_META,
                "num_params": 126886696,
                "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#wide-resnet",
                "_metrics": {
                    "ImageNet-1K": {
                        "acc@1": 78.848,
                        "acc@5": 94.284,
                        }
                    },
                "_ops": 22.753,
                "_file_size": 485.357,
                "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
                },
            )
    DEFAULT = IMAGENET1K_V1