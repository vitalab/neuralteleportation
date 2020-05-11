from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel


def get_interpolated_weight(model: NeuralTeleportationModel, teleported_model: NeuralTeleportationModel, scale: float):
    assert 0 < scale <= 1
    interpolated_model = NeuralTeleportationModel();
    # interpolated_model = teleported_model.copy()
    interpolated_model.set_weights((1-scale)*model.get_weights() + NeuralTeleportationModel().get_weights()*scale)
    interpolated_model.
    model.get_weights()
    return interpolated_model