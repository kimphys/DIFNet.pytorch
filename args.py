class args():

    # hyperparameters
    put_type = 'mean'
    balance = 0.01

    # training args
    epochs = 100 # "number of training epochs, default is 2"
    save_per_epoch = 1
    batch_size = 1 # "batch size for training, default is 4"
    dataset1 = "./train_visible.txt"
    dataset2 = "./train_lwir.txt"
    HEIGHT = 256
    WIDTH = 256
    lr = 1e-4 # "Initial learning rate, default is 0.0001"
    lr_step = 10 # Learning rate is halved in 10 epochs 	
    resume = "./models/sample.pt" # if you have, please put the path of the model like "./models/densefuse_gray.model"
    # resume = None
    save_model_dir = "./models/" #"path to folder where trained model with checkpoints will be saved."
    workers = 20

    # For GPU training
    world_size = -1
    rank = -1
    dist_backend = 'nccl'
    gpu = 0,1,2,3
    multiprocessing_distributed = True
    distributed = None

    # For testing
    test_save_dir = "./"
    test_visible = "./test_visible.txt"
    test_lwir = "./test_lwir.txt"
