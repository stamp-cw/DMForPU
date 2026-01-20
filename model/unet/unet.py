from selector.model_selector import register_model
from diffusers import UNet2DConditionModel
@register_model(name='UNet')
class UNet(UNet2DConditionModel):
    def __init__(self, config):
        super(UNet, self).__init__()
        n_channels = config.model.in_channels
        n_classes = config.model.out_channels
        bilinear = config.model.bilinear
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear