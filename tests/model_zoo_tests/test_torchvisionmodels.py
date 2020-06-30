from tests.model_test import test_teleport


class TestTorchVisionModels:
    @property
    def models_functions(self):
        raise NotImplementedError

    @property
    def default_input_shape(self):
        raise NotImplementedError

    def test_models_teleportation(self):
        for model_fn in self.models_functions:
            model = model_fn(num_classes=10)
            test_teleport(model, input_shape=self.default_input_shape, model_name=model_fn.__name__)

            assert True

    def test_pre_trained_models(self):
        """
        Test if the model can correctly load the pretrained weights.
        If not, there is an error in the model construction.
        """

        for model_fn in self.models_functions:
            model = model_fn(pretrained=True, num_classes=1000)  # Test if the model can load weights correctly

        assert True

