from pathlib import Path

from tqdm import tqdm

import torch
from torch import nn


def test_(model, val_dataloader, args, mean, std, epoch=0):
    model.eval()

    if args.local_rank == 0:
        file = open(f"results_{epoch}.csv", 'w')
        file.write("Id,Label\n")

    with torch.no_grad():
        for nbatch, data in enumerate(tqdm(val_dataloader)):
            img = data['image']
            img = torch.Tensor(img).cuda()
            img.sub_(mean).div_(std)

            images_path = data['image_path']

            output = model(img)

            predictions = 1 - nn.functional.softmax(output, dim=1).data.cpu().numpy()[:, 0]

            if args.local_rank == 0:
                for (image_path, prediction) in zip(images_path, predictions):
                    file.write("{},{}\n".format(Path(image_path).parts[-1], prediction.item()))

    if args.local_rank == 0:
        file.close()


def load_checkpoint(model, checkpoint):
    """
    Load model from checkpoint.
    """
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    saved_model = od["model"]
    model.load_state_dict(saved_model)
