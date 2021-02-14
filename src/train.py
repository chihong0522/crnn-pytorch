import os

import cv2
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import MSELoss

from dataset import TUM_Dataset
from model import CRNN
from evaluate import evaluate
from config import train_config as config
from tqdm import tqdm


def train_batch(crnn, data, optimizer, criterion, device, batch_size):
    crnn.hidden = crnn.init_hidden(batch_size=batch_size)
    crnn.train()

    previous_images, current_images, targets = [d.to(device) for d in data]
    prediction = crnn(previous_images, current_images)
    loss = criterion(prediction,targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    val_batch_size = config['val_batch_size']
    lr = config['lr']
    show_interval = config['show_interval']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']
    valid_max_iter = config['valid_max_iter']

    img_width = config['img_width']
    img_height = config['img_height']
    data_dir = config['data_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Curret device: {device}')

    train_dataset = TUM_Dataset(split='train')
    valid_dataset = TUM_Dataset(split='val')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=cpu_workers)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=cpu_workers)

    num_class = 7
    crnn = CRNN(3, img_height, img_width, num_class, train_batch_size,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = MSELoss()
    criterion.to(device)

    assert save_interval % valid_interval == 0
    
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0

        pbar_total = len(train_loader)
        pbar = tqdm(total=pbar_total, desc="Train")

        for train_data in train_loader:
            train_size = train_data[0].size(0)
            loss = train_batch(crnn, train_data, optimizer, criterion, device, train_size)
        
            tot_train_loss += loss
            tot_train_count += train_size
            pbar.update(1)
        pbar.close()
        print(f'trainning loss: {tot_train_loss / tot_train_count}')
        eval_loss = evaluate(crnn, valid_loader, criterion)
        print(f'validation loss: {eval_loss}')

        if epoch % 10 == 0:
            prefix = 'crnn'
            save_model_path = os.path.join(config['checkpoints_dir'],
                                            f'{prefix}_{epoch:06}_evalLoss_{eval_loss}.pt')
            torch.save(crnn.state_dict(), save_model_path)
            print('save model at ', save_model_path)

        


if __name__ == '__main__':
    main()
