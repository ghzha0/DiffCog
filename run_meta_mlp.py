"""
Train a diffusion model for cognitive diagnosis
"""
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import models.gaussian_diffusion as gd
from models.DNN import DNN, GCN
from models.Decoder import *
from models.Encoder import MLP_Encoder
from models.ConstraintModel import ConstraintModel
import evaluate_utils
import data_utils
from scipy import sparse as sp
import tqdm
import logging
import argparse
import json


def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='参数位置', default='config.yml')
    return parser.parse_args()


args = arg_parse()
print(f'./config/{args.config}')
with open(f'./config/{args.config}', 'r') as file:
    config = yaml.safe_load(file)['DiffCog']

logging.basicConfig(filename=config['log_name'], level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
print(config)
logging.info(config)

os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu'])
device = torch.device("cuda:0" if config['cuda'] else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

# DATA LOAD
train_path = config['data_path'] + 'train_list.npy'
valid_path = config['data_path'] + 'valid_list.npy'
test_path = config['data_path'] + 'test_list.npy'
concept_path = config['data_path'] + 'concept_map.json'
kg_path = config['data_path'] + 'kg_mat.npy'

dataset_info = yaml.full_load(open(config['data_path'] + 'info_filtered.yml'))
print(dataset_info)

meta_list, noised_list, train_matrix, valid_list, test_list, concept_map, kg_mat = data_utils.meta_mlp_cog_data_load(
    config['meta_ratio'],
    config['noise_ratio'],
    train_path, valid_path,
    test_path,
    concept_path, kg_path, dataset_info)

np.save(f"./Visualization/train_matrix_{config['noise_ratio']}.npy", train_matrix)
train_matrix = torch.Tensor(train_matrix).to(device)
train_kg = torch.Tensor(kg_mat > config['relation_value']).float().to(device)
meta_dataset = data_utils.DataCog(meta_list, concept_map, know_num=dataset_info['kc_all'])
noised_dataset = data_utils.DataCog(noised_list, concept_map, know_num=dataset_info['kc_all'])
valid_dataset = data_utils.DataCog(valid_list, concept_map, know_num=dataset_info['kc_all'])
test_dataset = data_utils.DataCog(test_list, concept_map, know_num=dataset_info['kc_all'])
meta_loader = DataLoader(meta_dataset, batch_size=config['batch_size'], pin_memory=True,
                          shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
noised_loader = DataLoader(noised_dataset, batch_size=config['batch_size'], pin_memory=True,
                          shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
print('data ready.')
logging.info('data ready.')


def cog_evaluate(data_loader, train_matrix):
    model.eval()
    enc.eval()
    dec.eval()
    predict_items = []
    target_items = []
    with torch.no_grad():
        for batch_idx, batch in tqdm.tqdm(enumerate(data_loader)):
            triple, concept = batch
            user, item, score = triple[:, 0], triple[:, 1], triple[:, 2]
            user_vector = train_matrix[user.tolist(), :]
            item_vector = train_matrix[:, item.tolist()].T
            user, item, score, concept = user.to(device), item.to(device), score.to(device), concept.to(device)
            user_emb, item_emb, steps = enc(user_vector, item_vector)
            if config['is_diffusion']:
                if config['is_adaptive_step']:
                    user_emb_diff = model(user_emb, steps)
                else:
                    user_emb_diff = diffusion.p_sample(model, user_emb, config['sampling_steps'],
                                                       config['sampling_noise'])
            else:
                user_emb_diff = user_emb
            predictions = dec(user_emb_diff, item_emb, concept).view(-1).detach().cpu().tolist()
            targets = score.detach().cpu().tolist()
            predict_items.extend(predictions)
            target_items.extend(targets)
    test_results = evaluate_utils.computeAccuracy(target_items, predict_items)
    return test_results


final_acc = []
final_auc = []
final_f1 = []
for seed in config['random_seed']:
    random_seed = seed
    torch.manual_seed(random_seed)  # cpu
    torch.cuda.manual_seed(random_seed)  # gpu
    np.random.seed(random_seed)  # numpy
    random.seed(random_seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn

    # Build Gaussian Diffusion
    if config['mean_type'] == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif config['mean_type'] == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % config['mean_type'])

    diffusion = gd.GaussianDiffusion(mean_type, config['noise_schedule'],
                                     config['noise_scale'], config['noise_min'],
                                     config['noise_max'], config['steps'], device).to(device)

    # BUILD MLP
    if config['is_constraint']:
        model = DNN(dataset_info['kc_all'], config['hid_dim'], config['emb_size'], time_type="cat", norm=config['norm'],
                    adj=train_kg).to(
            device)
    else:
        model = DNN(dataset_info['kc_all'], config['hid_dim'], config['emb_size'], time_type="cat",
                    norm=config['norm']).to(device)
    enc = MLP_Encoder(user_num=dataset_info['stu_all'], exer_num=dataset_info['exer_all'],
                      know_num=dataset_info['kc_all'],
                      embed_dim=config['enc_embed_dim']).to(device)

    if config['decoder'] == 'NCD':
        print('decoder is NCD!')
        logging.info('decoder is NCD!')
        dec = MLP_Decoder(know_num=dataset_info['kc_all'], embed_dim=config['dec_embed_dim']).to(device)
    elif config['decoder'] == 'IRT':
        print('decoder is IRT!')
        logging.info('decoder is IRT!')
        dec = IRT_Decoder(know_num=dataset_info['kc_all']).to(device)
    elif config['decoder'] == 'MIRT':
        print('decoder is MIRT!')
        logging.info('decoder is MIRT!')
        dec = MIRT_Decoder(know_num=dataset_info['kc_all'], latent_dim=config['latent_dim']).to(device)
    else:
        raise NotImplementedError("This decoder is not yet implemented!")

    optimizer = optim.Adam(model.parameters(), lr=config['diff_lr'])
    enc_opt = optim.Adam([
            {'params': enc.user_enc.parameters(), 'lr': config['lr']},
            {'params': enc.item_enc.parameters(), 'lr': config['lr']}
        ]
    )
    meta_opt = optim.Adam(enc.time_enc.parameters(), lr=config['lr'])
    dec_opt = optim.Adam(dec.parameters(), lr=config['lr'])
    print("models ready.")
    logging.info("models ready.")

    param_num = 0
    mlp_num = sum([param.nelement() for param in model.parameters()])
    enc_num = sum([param.nelement() for param in enc.parameters()])
    dec_num = sum([param.nelement() for param in dec.parameters()])
    diff_num = sum([param.nelement() for param in diffusion.parameters()])
    param_num = mlp_num + diff_num + enc_num + dec_num
    print("Number of all parameters:", param_num)
    logging.info(f"Number of all parameters: {param_num}")

    best_auc, best_epoch = -100, 0
    best_test_result = None
    print("Start training...")
    logging.info("Start training...")
    for epoch in range(1, config['epochs'] + 1):
        if epoch - best_epoch >= 10:
            print('-' * 18)
            print('Exiting from training early')
            logging.info('Exiting from training early')
            break

        model.train()
        enc.train()
        dec.train()
        start_time = time.time()

        batch_count = 0
        total_loss = 0.0
        total_vlbo = 0.0
        total_recon = 0.0
        total_recon_diff = 0.0
        loss_func = nn.BCELoss()

        start_index = 0
        batch_size = config['batch_size']

        target_items = []
        predict_items = []

        # noised train
        # freeze time_enc的参数
        enc.time_enc.require_grad = True
        enc.user_enc.require_grad = True
        enc.item_enc.require_grad = True
        dec.require_grad = True
        model.require_grad = True
        print('noise training!')
        for batch_idx, batch in tqdm.tqdm(enumerate(noised_loader)):
            triple, concept = batch
            user, item, score = triple[:, 0], triple[:, 1], triple[:, 2]
            user_vector = train_matrix[user.tolist(), :]
            item_vector = train_matrix[:, item.tolist()].T
            user, item, score, concept = user.to(device), item.to(device), score.to(device), concept.to(device)
            user_emb, item_emb, steps = enc(user_vector, item_vector)
            vlbo = 0
            recon_loss_diff = 0

            if config['is_diffusion']:
                terms = diffusion.training_losses(model, user_emb, config['reweight'])
                vlbo = terms["loss"].mean()
                # enc and dec loss
                if config['is_adaptive_step']:
                    user_emb_diff = model(user_emb, steps)
                else:
                    user_emb_diff = user_emb
                batch_recon_diff = dec(user_emb_diff, item_emb, concept)
                recon_loss_diff = loss_func(batch_recon_diff.view(-1), score.float())

                # optimizer initialization
                optimizer.zero_grad()
                enc_opt.zero_grad()
                dec_opt.zero_grad()
                loss = vlbo + recon_loss_diff
                loss.backward()
                optimizer.step()
                enc_opt.step()
                dec_opt.step()
            else:
                batch_recon = dec(user_emb, item_emb, concept)
                recon_loss_diff = loss_func(batch_recon.view(-1), score.float())
                # optimizer initialization
                enc_opt.zero_grad()
                dec_opt.zero_grad()
                loss = vlbo + recon_loss_diff
                loss.backward()
                enc_opt.step()
                dec_opt.step()
            total_vlbo += vlbo.item()
            total_recon_diff += recon_loss_diff.item()
            total_loss += loss.item()

        total_loss /= (batch_idx + 1)
        total_vlbo /= (batch_idx + 1)
        total_recon /= (batch_idx + 1)
        total_recon_diff /= (batch_idx + 1)

        print(
            "Noised Runing Epoch {:03d} ".format(epoch) +
            ' vlbo {:.4f} '.format(total_vlbo) +
            ' recon {:.4f} '.format(total_recon) +
            ' recon diff {:.4f} '.format(total_recon_diff) +
            ' train loss {:.4f} '.format(total_loss) +
            " costs " + time.strftime(
                "%H: %M: %S", time.gmtime(time.time() - start_time))
        )
        print('---' * 18)

        logging.info(
            "Noised Runing Epoch {:03d} ".format(epoch) +
            ' vlbo {:.4f} '.format(total_vlbo) +
            ' recon {:.4f} '.format(total_recon) +
            ' recon diff {:.4f} '.format(total_recon_diff) +
            ' train loss {:.4f} '.format(total_loss) +
            " costs " + time.strftime(
                "%H: %M: %S", time.gmtime(time.time() - start_time))
        )
        logging.info('---' * 18)

        total_loss = 0.0
        total_vlbo = 0.0
        total_recon = 0.0
        total_recon_diff = 0.0

        # clean train
        enc.time_enc.require_grad = True
        enc.user_enc.require_grad = False
        enc.item_enc.require_grad = False
        dec.require_grad = False
        model.require_grad = False
        print('meta training!')
        for _ in range(config['meta_steps']):
            for batch_idx, batch in tqdm.tqdm(enumerate(meta_loader)):
                triple, concept = batch
                user, item, score = triple[:, 0], triple[:, 1], triple[:, 2]
                user_vector = train_matrix[user.tolist(), :]
                item_vector = train_matrix[:, item.tolist()].T
                user, item, score, concept = user.to(device), item.to(device), score.to(device), concept.to(device)
                user_emb, item_emb, steps = enc(user_vector, item_vector)
                vlbo = 0
                recon_loss_diff = 0

                if config['is_diffusion']:
                    # enc and dec loss
                    if config['is_adaptive_step']:
                        user_emb_diff = model(user_emb, steps)
                    else:
                        user_emb_diff = user_emb
                    batch_recon_diff = dec(user_emb_diff, item_emb, concept)
                    recon_loss_diff = loss_func(batch_recon_diff.view(-1), score.float())

                    # optimizer initialization
                    meta_opt.zero_grad()
                    loss = recon_loss_diff
                    loss.backward()
                    meta_opt.step()
                else:
                    batch_recon = dec(user_emb, item_emb, concept)
                    recon_loss_diff = loss_func(batch_recon.view(-1), score.float())

                    # optimizer initialization
                    meta_opt.zero_grad()
                    loss = recon_loss_diff
                    loss.backward()
                    meta_opt.step()
                total_recon_diff += recon_loss_diff.item()
            total_recon_diff /= (batch_idx + 1)

            print(
                "Meta Runing Epoch {:03d} ".format(epoch) +
                ' vlbo {:.4f} '.format(total_vlbo) +
                ' recon {:.4f} '.format(total_recon) +
                ' recon diff {:.4f} '.format(total_recon_diff) +
                ' train loss {:.4f} '.format(total_loss) +
                " costs " + time.strftime(
                    "%H: %M: %S", time.gmtime(time.time() - start_time))
            )
            print('---' * 18)

            logging.info(
                "Meta Runing Epoch {:03d} ".format(epoch) +
                ' vlbo {:.4f} '.format(total_vlbo) +
                ' recon {:.4f} '.format(total_recon) +
                ' recon diff {:.4f} '.format(total_recon_diff) +
                ' train loss {:.4f} '.format(total_loss) +
                " costs " + time.strftime(
                    "%H: %M: %S", time.gmtime(time.time() - start_time))
            )
            logging.info('---' * 18)

        if epoch % 1 == 0:
            valid_acc, valid_auc, valid_f1, valid_rmse = cog_evaluate(valid_loader, train_matrix)
            test_acc, test_auc, test_f1, test_rmse = cog_evaluate(test_loader, train_matrix)
            print(f'valid results: acc:{valid_acc} auc:{valid_auc} f1:{valid_f1} rmse:{valid_rmse}')
            print(f'test results: acc:{test_acc} auc:{test_auc} f1:{test_f1} rmse:{test_rmse}')
            logging.info(f'valid results: acc:{valid_acc} auc:{valid_auc} f1:{valid_f1} rmse:{valid_rmse}')
            logging.info(f'test results: acc:{test_acc} auc:{test_auc} f1:{test_f1} rmse:{test_rmse}')

            if valid_auc > best_auc:
                best_auc, best_epoch = valid_auc, epoch
                best_valid_acc, best_valid_auc, best_valid_f1, best_valid_rmse = valid_acc, valid_auc, valid_f1, valid_rmse
                best_test_acc, best_test_auc, best_test_f1, best_test_rmse = test_acc, test_auc, test_f1, test_rmse

                if not os.path.exists(config['save_path']):
                    os.makedirs(config['save_path'])
                torch.save(enc, f"./Visualization/{config['dataset']}_{config['decoder']}_{config['noise_ratio']}.pth")

    final_acc.append(best_test_acc)
    final_auc.append(best_test_auc)
    final_f1.append(best_test_f1)

    print('===' * 18)
    print("End. Best Epoch {:03d} ".format(best_epoch))
    print(f'valid results: acc:{best_valid_acc} auc:{best_valid_auc} f1:{best_valid_f1} rmse:{best_valid_rmse}')
    print(f'test results: acc:{best_test_acc} auc:{best_test_auc} f1:{best_test_f1} rmse:{best_test_rmse}')
    print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    logging.info('===' * 18)
    logging.info("End. Best Epoch {:03d} ".format(best_epoch))
    logging.info(f'valid results: acc:{best_valid_acc} auc:{best_valid_auc} f1:{best_valid_f1} rmse:{best_valid_rmse}')
    logging.info(f'test results: acc:{best_test_acc} auc:{best_test_auc} f1:{best_test_f1} rmse:{best_test_rmse}')

print('===' * 18)
print(f'test results: acc:{final_acc} auc:{final_auc} f1:{final_f1}')
print(f'test results: mean acc:{np.mean(final_acc)} mean auc:{np.mean(final_auc)} f1:{np.mean(final_f1)}')
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

logging.info('===' * 18)
logging.info(f'test results: acc:{final_acc} auc:{final_auc} f1:{final_f1}')
logging.info(f'test results: mean acc:{np.mean(final_acc)} mean auc:{np.mean(final_auc)} f1:{np.mean(final_f1)}')
logging.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
