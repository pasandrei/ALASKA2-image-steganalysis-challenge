from src.metrics import alaska_weighted_auc
from tqdm import tqdm

import torch
from torch import nn


def evaluate(model, val_dataloader, args, mean, std, loss_func):
    model.eval()

    complete_ground_truths = []
    complete_predictions = []

    total_loss = 0

    with torch.no_grad():
        for nbatch, data in enumerate(tqdm(val_dataloader)):
            img = data['image']
            img = torch.Tensor(img).cuda()
            img.sub_(mean).div_(std)

            ground_truth = data['ground_truth']

            output = model(img)

            loss = loss_func(output, ground_truth.cuda())
            total_loss += loss.item()

            predictions = 1 - nn.functional.softmax(output, dim=1).data.cpu().numpy()[:, 0]

            ground_truth[ground_truth != 0] = 1  # When using multiclass
            complete_ground_truths.extend(ground_truth.tolist())
            complete_predictions.extend(predictions.tolist())

        print("Average loss this val epoch: {}".format(total_loss / len(val_dataloader)))
        print(alaska_weighted_auc(complete_ground_truths, complete_predictions))


def load_checkpoint(model, checkpoint):
    """
    Load model from checkpoint.
    """
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    saved_model = od["model"]
    model.load_state_dict(saved_model)
