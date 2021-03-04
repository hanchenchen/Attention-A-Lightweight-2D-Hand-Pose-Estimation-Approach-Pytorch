import sys
import os
currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)
# NSRMhand-master
import shutil
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import model
import dataset
from src.utils import set_logger, update_lr, get_pck_with_sigma, get_pred_coordinates, save_images, save_limb_images
from src import loss

# ***********************  Parameter  ***********************

parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='config file for the experiment')
parser.add_argument('--GPU', type=int, default=0,
                    help='GPU. ')
args = parser.parse_args()
configs = json.load(open('configs/' + args.config_file))

target_sigma_list = [0.04, 0.06, 0.08, 0.1, 0.12]
select_sigma = 0.1

model_name = 'EXP_' + configs["name"]
save_dir = os.path.join(model_name, 'checkpoint/')
test_pck_dir = os.path.join(model_name, 'test.py')

os.makedirs(save_dir, exist_ok=True)
os.makedirs(test_pck_dir, exist_ok=True)

shutil.copy('configs/' + args.config_file, model_name)

# training parameters ****************************
data_root = configs["data_root"]
learning_rate = configs["learning_rate"]
batch_size = configs["batch_size"]
epochs = configs["epochs"]

# data parameters ****************************

device_ids = [args.GPU]      # multi-GPU
torch.cuda.set_device(device_ids[0])
cuda = torch.cuda.is_available()

logger = set_logger(os.path.join(model_name, 'train.log'))
logger.info("************** Experiment Name: {} **************".format(model_name))

# ******************** build model ********************
logger.info("Create Model ...")

model = model.light_Model(configs)
if cuda:
    model = model.cuda(device_ids[0])
    # cmodel = nn.DataParallel(model, device_ids=device_ids)

# ******************** data preparation  ********************
my_dataset = getattr(dataset, configs["dataset"])
train_data = my_dataset(data_root=data_root, mode='train')
valid_data = my_dataset(data_root=data_root, mode='valid')
test_data = my_dataset(data_root=data_root, mode='test')
logger.info('Total images in training data is {}'.format(len(train_data)))
logger.info('Total images in validation data is {}'.format(len(valid_data)))
logger.info('Total images in testing data is {}'.format(len(test_data)))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# ********************  ********************
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.0)

def train():
    logger.info('\nStart training ===========================================>')
    best_epo = -1
    max_pck = -1
    cur_lr = learning_rate

    logger.info('Initial learning Rate: {}'.format(learning_rate))

    for epoch in range(1, epochs + 1):
        logger.info('Epoch[{}/{}] ==============>'.format(epoch, epochs))
        model.train()
        train_label_loss = []

        for step, (img, label_terget, img_name, w, h) in enumerate(train_loader):
            # *************** target prepare ***************
            if cuda:
                img = img.cuda()
                label_terget = label_terget.cuda()
            # logger.info('LearningRate: {}'.format(optimizer.param_groups[0]['lr']))
            optimizer.zero_grad()
            label_pred = model(img)
            # limb_pred (FloatTensor.cuda) size:(bz,3,C,46,46)  after sigmoid
            # cm_pred   (FloatTensor.cuda) size:(bz,3,21,46,46)

            # *************** calculate loss ***************

            label_loss = loss.sum_mse_loss(label_pred.float(), label_terget.float())     # keypoint confidence loss
            label_loss.backward()
            optimizer.step()

            train_label_loss.append(label_loss.item())

            if step % 50 == 0:
                logger.info('STEP: {}  LOSS {}'.format(step, label_loss.item()))

        # *************** save sample image after one epoch ***************
        # save_images(cm_target[:, -1, ...].cpu(), cm_pred[:, -1, ...].cpu(),
        #             epoch, img_name, save_dir)
        #
        # save_limb_images(limb_target[:, -1, ...].cpu(), limb_pred[:, -1, ...].cpu(),
        #             epoch, img_name, save_dir)

        # *************** eval model after one epoch ***************
        eval_loss, cur_pck = eval(epoch, mode='valid')
        logger.info('EPOCH {} VALID PCK  {}'.format(epoch, cur_pck))
        logger.info('EPOCH {} TRAIN_LOSS {}'.format(epoch, sum(train_label_loss) / len(train_label_loss)))
        logger.info('EPOCH {} VALID_LOSS {}'.format(epoch, eval_loss))

        # *************** save current model and best model ***************
        if cur_pck > max_pck:
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            best_epo = epoch
            max_pck = cur_pck
        logger.info('Current Best EPOCH is : {}, PCK is : {}\n**************\n'.format(best_epo, max_pck))

        # save current model
        torch.save(model.state_dict(), os.path.join(save_dir, 'final_epoch.pth'))


    logger.info('Train Done! ')
    logger.info('Best epoch is {}'.format(best_epo))
    logger.info('Best Valid PCK is {}'.format(max_pck))



def eval(epoch, mode='valid'):
    if mode == 'valid':
        loader = valid_loader
        gt_labels = valid_data.all_labels
    else:
        loader = test_loader
        gt_labels = test_data.all_labels

    with torch.no_grad():
        all_pred_labels = {}        # save predict results
        eval_loss = []
        model.eval()
        for step, (img, label_terget, img_name, w, h) in enumerate(loader):
            if cuda:
                img = img.cuda()
            cm_pred = model(img)
            # limb_pred (FloatTensor.cuda) size:(bz,3,C,46,46)
            # cm_pred   (FloatTensor.cuda) size:(bz,3,21,46,46)

            all_pred_labels = get_pred_coordinates(cm_pred.cpu(),
                                                      img_name, w, h, all_pred_labels)
            loss_final = loss.sum_mse_loss(cm_pred.cpu(), label_terget)
            eval_loss.append(loss_final)

        # ******** save predict labels for valid/test data ********
        if mode == 'valid':
            pred_save_dir = os.path.join(save_dir, 'e' + str(epoch) + '_val_pred.json')
        else:
            pred_save_dir = os.path.join(test_pck_dir, 'test_pred.json')
        json.dump(all_pred_labels, open(pred_save_dir, 'w'), sort_keys=True, indent=4)

        # ************* calculate and save PCKs  ************
        pck_dict = get_pck_with_sigma(all_pred_labels, gt_labels, target_sigma_list)

        if mode == 'valid':
            pck_save_dir = os.path.join(save_dir, 'e' + str(epoch) + '_pck.json')
        else:
            pck_save_dir = os.path.join(test_pck_dir, 'pck.json')
        json.dump(pck_dict, open(pck_save_dir, 'w'), sort_keys=True, indent=4)

        select_pck = pck_dict[select_sigma]
        eval_loss = sum(eval_loss)/len(eval_loss)
    return eval_loss, select_pck



train()

logger.info('\nTESTING ============================>')
logger.info('Load Trained model !!!')
state_dict = torch.load(os.path.join(save_dir, 'best_model.pth'))
model.load_state_dict(state_dict)
eval(0, mode='test')

logger.info('Done!')




