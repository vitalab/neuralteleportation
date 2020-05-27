import unittest
from unittest import TestCase

from neuralteleportation.models.model_zoo.vggcob import *
from tests.model_zoo_tests.test_torchvisionmodels import TestTorchVisionModels


class TestVGGCOB(TestCase, TestTorchVisionModels):
    models_functions = [vgg11COB, vgg11_bnCOB]
    default_input_shape = (1, 3, 224, 224)


if __name__ == '__main__':
    unittest.main()
