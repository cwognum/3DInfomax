import os
import re

import click
import dgl
import fsspec
import numpy as np
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from infomax.datasets.qm9_inference_dataset import QM9InferenceDataset
from infomax.models import *


def load_model(model_type, model_parameters, checkpoint, avg_degree, device):
    model = globals()[model_type](avg_d=avg_degree, device=device, **model_parameters)
    checkpoint = torch.load(checkpoint)

    # get all the weights that have something from 'args.transfer_layers' in their keys name
    # but only if they do not contain 'teacher' and remove 'student.' which we need for loading from BYOLWrapper
    weights_key = 'model_state_dict'
    pretrained_gnn_dict = {
        re.sub('^gnn\.|^gnn2\.', 'node_gnn.', k.replace('student.', '')): v
        for k, v in checkpoint[weights_key].items()
        if any(transfer_layer in k for transfer_layer in ["gnn"])
           and 'teacher' not in k
           and not any(to_exclude in k for to_exclude in ["batch_norm"])
    }

    model_state_dict = model.state_dict()
    model_state_dict.update(pretrained_gnn_dict)  # update the gnn layers with the pretrained weights
    model.load_state_dict(model_state_dict)
    model.to(device)
    return model


@click.command()
@click.option("--input-smiles", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(file_okay=False), required=False)
@click.option("--limit-size", type=int, required=False)
@click.option("--num-layers-to-drop", type=int, default=1, required=False)
def cli(input_smiles, output_dir, limit_size, num_layers_to_drop):

    if output_dir is None:
        output_dir = os.path.dirname(input_smiles)

    smiles = np.load(input_smiles)[:limit_size]

    with fsspec.open("configs_clean/tune_QM9_homo.yml") as fd:
        args = yaml.safe_load(fd)

    device = torch.device("cpu")
    dataset = QM9InferenceDataset(smiles, device=device, reprocess=False)
    model = load_model(
        args["model_type"],
        args["model_parameters"],
        args["pretrain_checkpoint"],
        dataset.avg_degree,
        device
    )

    batch_size = 32
    dataloader = DataLoader(dataset, collate_fn=dgl.batch, batch_size=batch_size)
    for batch in tqdm.tqdm(dataloader, desc="Computing fingerprints", total=len(dataset) // batch_size):
        fps = model(batch, num_layers_to_drop=num_layers_to_drop)

    print(fps)

if __name__ == "__main__":
    cli()
