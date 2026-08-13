"""Model construction for the MobileNetV3 transfer-learning pipeline.

Centralizes the classifier-surgery / freezing snippets duplicated across
notebooks/03_mobilenet_transfer_learning.ipynb and notebooks/04_inference.ipynb,
so src/train.py, src/api.py and any future script share one implementation.
"""
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_model(num_classes, dropout, pretrained=False):
    """Build a MobileNetV3-Large with its classifier head resized to `num_classes`.

    Args:
        num_classes: number of output classes.
        dropout: dropout probability for the classifier's second-to-last
            layer (replaces the pretrained default of 0.2).
        pretrained: if True, download ImageNet weights (for training from
            scratch). If False (default), weights=None - use this for
            inference/serving, where load_state_dict() overwrites every
            weight anyway, so downloading the pretrained checkpoint would
            just waste bandwidth and startup time.
    """
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    model.classifier[2] = nn.Dropout(p=dropout)
    model.classifier[3] = nn.Linear(
        in_features=model.classifier[3].in_features,
        out_features=num_classes,
    )
    return model


def freeze_backbone(model, unfreeze_from_block):
    """Freeze every parameter, then unfreeze model.features[unfreeze_from_block:]
    and the whole classifier head. Mirrors the corresponding cell in
    notebooks/03_mobilenet_transfer_learning.ipynb."""
    for param in model.parameters():
        param.requires_grad = False

    for block in model.features[unfreeze_from_block:]:
        for param in block.parameters():
            param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def freeze_bn_in_frozen_blocks(model, unfreeze_from_block):
    """Keep BatchNorm layers inside the still-frozen backbone blocks in
    eval() even after model.train() is called elsewhere (e.g. at the start of
    every training epoch) - otherwise their running stats keep drifting away
    from the pretrained ImageNet statistics even though their weights don't
    receive gradients. See notebooks/03_mobilenet_transfer_learning.ipynb for
    the original discussion of why this matters."""
    trainable_ids = {id(m) for b in model.features[unfreeze_from_block:] for m in b.modules()}
    trainable_ids |= {id(m) for m in model.classifier.modules()}
    original_train = model.train

    def train_override(mode=True):
        original_train(mode)
        if mode:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d) and id(module) not in trainable_ids:
                    module.eval()
        return model

    model.train = train_override
    return model
