class MobileNet_Unfreezer():
    """
    Handles the freezing/unfreezing of the backbone of the model dynamically during training
    works for MobileNetV2
    """

    def __init__(self, unfreeze_epochs):
        self.unfreeze_epochs = unfreeze_epochs
        self.unfreeze_from_layer = [18, 17, 14, 11, 7, 4, 2, 1, 0]

    def freeze_backbone(self, model):
        for params in model.backbone.parameters():
            params.requires_grad = False

    def unfreeze_from(self, layer_idx, model):

        print("SA DAT UNFREEZE")
        for i in range(layer_idx, 19):
            for parameters in model.features[i].parameters():
                parameters.requires_grad = True

    def step(self, epoch, model):
        """
        uniformly unfreeze 15 layers from the start to the first decay
        unfreeze the rest of the layers as well at the second_decay
        if backbone was not frozen, this has no effect
        """

        for i in range(len(self.unfreeze_epochs)):
            if epoch == self.unfreeze_epochs[i]:
                self.unfreeze_from(self.unfreeze_from_layer[i], model)


class EfficientNet_Unfreezer():
    def __init__(self, unfreeze_epochs, unfreeze_from_layer):
        self.unfreeze_epochs = unfreeze_epochs
        self.unfreeze_from_layer = unfreeze_from_layer

    def unfreeze_from(self, layer_idx, model):
        print("SA DAT UNFREEZE")

        for i, parameters in enumerate(model.parameters()):
            if i >= layer_idx:
                parameters.requires_grad = True

    def step(self, epoch, model):
        """
        uniformly unfreeze 15 layers from the start to the first decay
        unfreeze the rest of the layers as well at the second_decay
        if backbone was not frozen, this has no effect
        """

        num_layers = 0
        for param in model.parameters():
            num_layers += 1

        for i in range(len(self.unfreeze_epochs)):
            if epoch == self.unfreeze_epochs[i]:
                self.unfreeze_from(int(self.unfreeze_from_layer[i] * num_layers), model)
