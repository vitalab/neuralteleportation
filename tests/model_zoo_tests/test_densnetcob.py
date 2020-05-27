import unittest
from unittest import TestCase

from neuralteleportation.models.model_zoo.densenetcob import *
from tests.model_zoo_tests.test_torchvisionmodels import TestTorchVisionModels


class TestDensenetCOB(TestCase, TestTorchVisionModels):
    models_functions = [densenet121COB]
    default_input_shape = (1, 3, 224, 224)


if __name__ == '__main__':
    unittest.main()
