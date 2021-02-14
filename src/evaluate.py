import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Synth90kDataset, synth90k_collate_fn
from model import CRNN
from ctc_decoder import ctc_decode
from config import evaluate_config as config

torch.backends.cudnn.enabled = False


def evaluate(crnn, dataloader, criterion):
    crnn.eval()

    tot_count = 0
    tot_loss = 0

    pbar_total = len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with torch.no_grad():
        for data in dataloader:
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            previous_images, current_images, targets = [d.to(device) for d in data]
            batch_size = images.size(0)
            crnn.hidden = crnn.init_hidden(batch_size=batch_size)
            prediction = crnn(previous_images, current_images)

            loss = criterion(prediction,targets)

            tot_count += batch_size
            tot_loss += loss.item()
            pbar.update(1)
        pbar.close()
    
    avg_eval_loss = tot_loss / tot_count
    return avg_eval_loss


def main():
    val_batch_size = config['val_batch_size']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']

    img_height = config['img_height']
    img_width = config['img_width']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    test_dataset = Synth90kDataset(root_dir=config['data_dir'], mode='test',
                                   img_height=img_height, img_width=img_width)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=cpu_workers,
        collate_fn=synth90k_collate_fn)

    num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    evaluation = evaluate(crnn, test_loader, criterion,
                          decode_method=config['decode_method'],
                          beam_size=config['beam_size'])
    print('test_evaluation: loss={loss}, acc={acc}'.format(**evaluation))


if __name__ == '__main__':
    main()
