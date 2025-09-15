##################################################
# Training Config
##################################################
workers = 1                 # number of Dataloader workers
epochs = 80                # number of epochs
batch_size = 8             # batch size
learning_rate = 8e-4        # initial learning rate

##################################################
# Model Config
##################################################
image_size = (256, 256)     # size of training images
net = 'resnet101'  # feature extractor
num_attentions = 32     # number of attention maps
beta = 5e-2                 # param for update feature centers

##################################################
# Dataset/Path Config
##################################################
tag = 'ndtwin'                # 'aircraft', 'bird', 'car', or 'dog'

# saving directory of .ckpt models
save_dir = './CelebA/wsdan-resnet101-cal/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
# ckpt = '/kaggle/input/celeba-448/model_last.pth'
# ckpt = save_dir + model_name
ckpt = '/kaggle/input/nd-twin-448/model_last_sept_15.pth'