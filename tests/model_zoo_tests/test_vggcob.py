import unittest
from unittest import TestCase

from neuralteleportation.models.model_zoo.vggcob import *
from tests.model_test import test_teleport
from tests.model_zoo_tests.test_torchvisionmodels import TestTorchVisionModels


class TestVGGCOB(TestCase, TestTorchVisionModels):
    models_functions = [vgg11COB, vgg11_bnCOB]
    default_input_shape = (1, 3, 224, 224)

    # def test_vgg11cob_teleportation(self):
    #     model = vgg11COB(pretrained=True)  # Test if the model can load weights correctly
    #
    #     model.eval()  # model must be set to eval because of dropout
    #     test_teleport(model, input_shape=(1, 3, 224, 224), model_name='vgg11')
    #
    #     assert True
    #
    # def test_vgg11bncob_teleportation(self):
    #     model = vgg11_bnCOB(pretrained=True)  # Test if the model can load weights correctly
    #
    #     model.eval()  # model must be set to eval because of dropout
    #     test_teleport(model, input_shape=(1, 3, 224, 224), model_name='vgg11_bn')
    #
    #     assert True


if __name__ == '__main__':
    unittest.main()
