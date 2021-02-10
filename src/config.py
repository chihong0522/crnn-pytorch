common_config = {
    'data_dir': 'data/mnt/ramdisk/max/90kDICT32px/',
    'img_width': 640,
    'img_height': 480,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
}

train_config = {
    'epochs': 10000,
    'train_batch_size': 8,
    'val_batch_size': 8,
    'lr': 0.0005,
    'show_interval': 10,
    'valid_interval': 500,
    'save_interval': 2000,
    'cpu_workers': 4,
    'reload_checkpoint': None,
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': 'checkpoints/'
}
train_config.update(common_config)

evaluate_config = {
    'val_batch_size': 8,
    'cpu_workers': 4,
    'reload_checkpoint': 'checkpoints/crnn_synth90k.pt',
    'decode_method': 'beam_search',
    'beam_size': 10,
}
evaluate_config.update(common_config)

path_config = {
    'train_dataset_dir': '/home/chihung/crnn-pytorch/data/rgbd_dataset_freiburg2_xyz/',
    'val_dataset_dir': '/home/chihung/crnn-pytorch/data/rgbd_dataset_freiburg1_xyz/'
}
