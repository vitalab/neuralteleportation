from torch.hub import load_state_dict_from_url
from torch.nn import Sequential, Module, init

from neuralteleportation.layers.activation import ReLU6COB
from neuralteleportation.layers.dropout import DropoutCOB
from neuralteleportation.layers.neuralteleportation import FlattenCOB
from neuralteleportation.layers.neuron import Conv2dCOB, LinearCOB, BatchNorm2dCOB
from neuralteleportation.layers.pooling import AdaptiveAvgPool2dCOB
from neuralteleportation.layers.merge import Add

__all__ = ['MobileNetV2COB', 'mobilenet_v2COB']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLUCOB(Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLUCOB, self).__init__(
            Conv2dCOB(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            BatchNorm2dCOB(out_planes),
            ReLU6COB(inplace=True)
        )


class InvertedResidualCOB(Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualCOB, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLUCOB(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLUCOB(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            Conv2dCOB(hidden_dim, oup, 1, 1, 0, bias=False),
            BatchNorm2dCOB(oup),
        ])
        self.conv = Sequential(*layers)
        self.add = Add()

    def forward(self, x):
        if self.use_res_connect:
            return self.add(x, self.conv(x))
        else:
            return self.conv(x)


class MobileNetV2COB(Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2COB, self).__init__()

        if block is None:
            block = InvertedResidualCOB
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLUCOB(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLUCOB(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = Sequential(*features)

        self.avgpool = AdaptiveAvgPool2dCOB((1, 1))
        self.flatten = FlattenCOB()

        # building classifier
        self.classifier = Sequential(
            DropoutCOB(0.2),
            LinearCOB(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, Conv2dCOB):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, BatchNorm2dCOB):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, LinearCOB):
                init.normal_(m.weight, 0, 0.01)
                init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2COB(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2COB(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    from tests.model_test import test_teleport
    from torchsummary import summary
    from torchvision.models import mobilenet_v2
    import torch

    original_model = mobilenet_v2(pretrained=True).eval()
    summary(original_model, (3, 224, 224), device='cpu')

    cob_mdoel = mobilenet_v2COB(pretrained=True).eval()

    summary(cob_mdoel, (3, 224, 224), device='cpu')

    x = torch.rand((1, 3, 224,224))

    pred1 = original_model(x)
    pred2 = cob_mdoel(x)
    diff_average = torch.mean(torch.abs((pred1 - pred2)))

    print("Test pretrained weightes: ")
    print("Original: ", pred1.flatten()[:10])
    print("COB implementation: ", pred2.flatten()[:10])
    print("Diff prediction average: ", diff_average)

    test_teleport(cob_mdoel, (1, 3, 224, 224), verbose=True, model_name='MobileNetV2COB')



