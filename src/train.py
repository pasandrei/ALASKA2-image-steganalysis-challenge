from tqdm import tqdm

import torch
try:
    from apex import amp
except ImportError:
    pass


def train_loop(model, loss_func, optimizer, train_dataloader, logger, args, mean, std, device):
    model.train()
    part_loss = total_loss = 0

    for param_group in optimizer.param_groups:
        print("Current learning rate:", param_group['lr'])

    print(len(train_dataloader))

    nr_batches = 0
    for nbatch, data in enumerate(tqdm(train_dataloader)):
        img = data['image']
        img = torch.Tensor(img).cuda()
        img.sub_(mean).div_(std)

        # ground_truth = data['ground_truth'].to(device).float()
        ground_truth = data['ground_truth'].cuda()

        # output = model(img).view(-1)
        output = model(img)

        loss = loss_func(output, ground_truth)
        total_loss += loss.item()
        part_loss += loss.item()

        if (nbatch+1) % (len(train_dataloader) // 10) == 0:
            print("Average loss this epoch: {}".format(part_loss / (len(train_dataloader) // 10)))
            part_loss = 0

        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        nr_batches = nbatch
        # if epoch == 1 and nbatch > len(train_dataloader)/10:
        #     break

    print("Average loss this epoch: {}".format(total_loss/len(train_dataloader)))

    return total_loss/len(train_dataloader)


def load_checkpoint(file_path, model, optimizer=None, scheduler=None):
    """
    Load model, optimizer, scheduler and the last epoch from checkpoint
    """
    checkpoint = torch.load(file_path, map_location="cuda:0")

    model.load_state_dict(checkpoint['model'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint.get('epoch', 0)
    print('Model loaded successfully')

    return model, optimizer, scheduler, start_epoch
