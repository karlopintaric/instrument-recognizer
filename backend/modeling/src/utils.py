from glob import glob
import yaml
import torch.optim as optim
from types import SimpleNamespace
from torchvision.transforms import Compose

CLASSES = ['tru', 'sax', 'vio', 'gac', 'org',
           'cla', 'flu', 'voi', 'gel', 'cel', 'pia']


def get_wav_files(base_path):
    return glob(f"{base_path}/**/*.wav", recursive=True)

def extract_from_df(df, cols):
    values = []
    for col in cols:
        v = df[col].iloc[0]
        values.append(v)
    return values


def parse_config(config_path):
    with open(config_path) as file:
        return SimpleNamespace(**yaml.safe_load(file))


def init_transforms(fn_dict, module):
    transforms = init_objs(fn_dict, module)
    if transforms is not None:
        transforms = ComposeTransforms(transforms)
    return transforms


def init_objs(fn_dict, module):

    if fn_dict is None:
        return None

    transforms = []
    for transform in fn_dict.keys():
        fn = getattr(module, transform)
        fn_args = fn_dict[transform]

        if fn_args is None:
            transforms.append(fn())
        else:
            transforms.append(fn(**fn_args))

    return transforms


def init_obj(fn_dict, module, *args, **kwargs):

    if fn_dict is None:
        return None

    name = list(fn_dict.keys())[0]

    fn = getattr(module, name)
    fn_args = fn_dict[name]

    if fn_args is not None:
        assert all([k not in fn_args for k in kwargs])
        fn_args.update(kwargs)

        return fn(*args, **fn_args)
    else:
        return fn(*args, **kwargs)


def LLRD(config, model):
    config = config.LLRD
    lr = config["base_lr"]
    optimizer_grouped_parameters = [{
        'params': [p for n, p in model.named_parameters() if not (("embeddings" in n) or ("encoder.layer" in n))],
        'lr': lr,
        "weight_decay": 0
    }]
    no_decay = ["bias", "layernorm"]
    # initialize lrs for every layer
    #num_layers = model.config.num_hidden_layers
    model_type = "audio_spectrogram_transformer"
    layers = [getattr(model.module.model, model_type).embeddings] + \
        list(getattr(model.module.model, model_type).encoder.layer)
    layers.reverse()

    weight_decay = config["weight_decay"]
    for layer in layers:
        lr *= config["lr_decay_rate"]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

    return optimizer_grouped_parameters


class ComposeTransforms:

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, input, *args):
        for t in self.transforms:
            input = t(input, *args)
        return input


def freeze(model):

    for param in model.parameters():
        param.requires_grad = False

    return model


def unfreeze(model):

    for param in model.parameters():
        param.requires_grad = True

    return model


if __name__ == "__main__":
    import models
    config = parse_config("./config.yaml")
    model = models.RNN(128, 64, 3, 11)
    optimizer = init_obj(config.optimizer, optim, model.parameters())
