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
@click.option("--batch-size", type=int, default=1, required=False)
@click.option("--process-dataset/--use-cached-dataset", default=False)
@click.option("--num-layers-to-drop", type=int, default=1, required=False)
def cli(input_smiles, output_dir, limit_size, batch_size, process_dataset, num_layers_to_drop):

    if output_dir is None:
        output_dir = os.path.dirname(input_smiles)

    smiles = np.load(input_smiles)[:limit_size]

    with fsspec.open("configs_clean/tune_QM9_homo.yml") as fd:
        args = yaml.safe_load(fd)

    device = torch.device("cpu")
    dataset = QM9InferenceDataset(smiles, device=device, reprocess=process_dataset)
    model = load_model(
        args["model_type"],
        args["model_parameters"],
        args["pretrain_checkpoint"],
        dataset.avg_degree,
        device
    )

    fps = []
    dataloader = DataLoader(dataset, collate_fn=dgl.batch, batch_size=batch_size)
    for idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Computing fingerprints", total=len(smiles) // batch_size)):
        try:
            fp = model(batch, num_layers_to_drop=num_layers_to_drop)
            fps.extend(fp.detach().tolist())
        except KeyboardInterrupt:
            raise
        except:
            # This is not ideal, but I'm not sure how to otherwise easily discover which element in the batch
            # causes the issue
            print(f"Encoding {batch} as all-zeroes")
            fps.extend([np.zeros_like(fps[0])] * batch_size)

    for idx in sorted(dataset.failed_indices):
        print(f"Encoding {smiles[idx] if smiles[idx] != '' else '<empty>'} as all-zeroes")
        fps.insert(idx, np.zeros_like(fps[0]))

    fps = np.array(fps)
    with fsspec.open(os.path.join(output_dir, f"QM9_fingerprints_{num_layers_to_drop}.npy"), 'wb') as fd:
        np.save(fd, fps)


if __name__ == "__main__":
    cli()
