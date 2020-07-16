from pathlib import Path

from configs.system_device import device
from tqdm import tqdm

import time
import torch
from torch import nn


def test_(model, val_dataloader, args, mean, std):
    model.eval()

    file = open(f"results_{args.local_rank}.csv", 'w')

    file.write("Id,Label\n")

    sigmoid = torch.nn.Sigmoid()

    for nbatch, data in enumerate(tqdm(val_dataloader)):
        img = data['image']
        img = torch.Tensor(img).cuda()
        img.sub_(mean).div_(std)

        images_path = data['image_path']

        output = model(img)
        # predictions = sigmoid(output)

        predictions = 1 - nn.functional.softmax(output, dim=1).data.cpu().numpy()[:, 0]

        for (image_path, prediction) in zip(images_path, predictions):
            file.write("{},{}\n".format(Path(image_path).parts[-1], prediction.item()))

    file.close()


# precision, recall, f1 = compute_metrics('dataset_test.csv', 'results.csv')
#
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 score: {f1}")
# print("====================\n")


def benchmark_inference_loop(model, val_dataloader, args, mean, std, logger):
    benchmark_warmup = 2
    benchmark_iterations = 8
    model.eval()

    softmax = torch.nn.Softmax(dim=1)
    start_time = None

    if args.backbone in ['mobilenetv2_3d', 'resnet18_3d']:
        sample_duration = 4
    else:
        sample_duration = 1

    for nbatch, data in enumerate(val_dataloader):
        if nbatch >= benchmark_warmup:
            start_time = time.time()

        img = data['image']
        img = torch.Tensor(img).to(device)
        img.sub_(mean).div_(std)

        output = model(img)
        predictions = softmax(output)

        if nbatch >= benchmark_warmup + benchmark_iterations:
            break

        if nbatch >= benchmark_warmup:
            logger.update(args.batch_size, sample_duration, time.time() - start_time)

    logger.print_result()


def load_checkpoint(model, checkpoint):
    """
    Load model from checkpoint.
    """
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    saved_model = od["model"]
    model.load_state_dict(saved_model)
