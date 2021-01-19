import unittest
from unittest import TestCase

from neuralteleportation.models.model_zoo.mobilenet import *
from tests.model_zoo_tests.test_torchvisionmodels import TestTorchVisionModels


class TestMobileNetCOB(TestCase, TestTorchVisionModels):
    models_functions = [mobilenet_v2COB]
    default_input_shape = (1, 3, 224, 224)


if __name__ == '__main__':
    unittest.main()
