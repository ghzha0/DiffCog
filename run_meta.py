import os
import sys
import yaml

noise_ratio = [0.15]
meta_steps = [1]
decoders = ['NCD']
ablation_types = [(True, True)]
datasets = ['assist0910']
meta_ratio = 0.15

for meta_step in meta_steps:
    for dataset in datasets:
        for nr in noise_ratio:
            for ablation_type in ablation_types:
                for decoder in decoders:
                    print(f"Decoder: {decoder}")
                    print(f"Noise_ratio is: {nr}")
                    with open('config/run_meta_config.yml', 'r') as file:
                        config = yaml.safe_load(file)
                    config['DiffCog']['random_seed'] = [2023]
                    config['DiffCog']['dataset'] = dataset
                    config['DiffCog']['data_path'] = f'../datasets/{dataset}/'
                    config['DiffCog']['batch_size'] = 512
                    if dataset == 'assist0910':
                        if decoder == 'NCD':
                            config['DiffCog']['enc_embed_dim'] = 256
                            config['DiffCog']['dec_embed_dim'] = 256
                            config['DiffCog']['hid_dim'] = 256
                        else:
                            config['DiffCog']['enc_embed_dim'] = 128
                            config['DiffCog']['dec_embed_dim'] = 128
                            config['DiffCog']['hid_dim'] = 256
                    elif dataset == 'assist1213':
                        if decoder == 'NCD':
                            config['DiffCog']['enc_embed_dim'] = 256
                            config['DiffCog']['dec_embed_dim'] = 256
                            config['DiffCog']['hid_dim'] = 512
                        else:
                            config['DiffCog']['enc_embed_dim'] = 256
                            config['DiffCog']['dec_embed_dim'] = 256
                            config['DiffCog']['hid_dim'] = 1000
                    else:
                        config['DiffCog']['enc_embed_dim'] = 256
                        config['DiffCog']['dec_embed_dim'] = 256
                        config['DiffCog']['hid_dim'] = 1000
                    config['DiffCog']['meta_steps'] = meta_step
                    config['DiffCog']['meta_ratio'] = meta_ratio
                    config['DiffCog']['noise_ratio'] = nr
                    config['DiffCog']['decoder'] = decoder
                    config['DiffCog']['gpu'] = 5
                    config['DiffCog']['is_diffusion'], config['DiffCog']['is_adaptive_step'] = ablation_type
                    config['DiffCog']['log_name'] = (
                            f"./log/{config['DiffCog']['dataset']}_" +
                            f"{config['DiffCog']['decoder']}_ " +
                            f"{config['DiffCog']['is_diffusion']}_" +
                            f"{config['DiffCog']['is_adaptive_step']}_" +
                            f"{config['DiffCog']['meta_steps']}_"
                            f"{config['DiffCog']['noise_ratio']}.log"
                    )
                    with open('config/run_meta_config.yml', 'w') as file:
                        yaml.safe_dump(config, file)
                    print("Ready to run!")
                    os.system(f"python run_meta_mlp.py --config run_meta_config.yml")
