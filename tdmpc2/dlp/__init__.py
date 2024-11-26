import json
import os

import torch

from dlp.models import ObjectDynamicsDLP


def create_ddlp(config_path, action_dim):
    with open(config_path, 'r') as f:
        config = json.load(f)

    # data and general
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    root = config['root']  # dataset root
    animation_horizon = config['animation_horizon']
    batch_size = config['batch_size']
    max_norm = config.get('max_norm', 0.5)
    lr = config['lr']
    num_epochs = config['num_epochs']
    topk = min(config['topk'], config['n_kp_enc'])  # top-k particles to plot
    eval_epoch_freq = config['eval_epoch_freq']
    weight_decay = config['weight_decay']
    iou_thresh = config['iou_thresh']  # threshold for NMS for plotting bounding boxes
    run_prefix = config['run_prefix']
    if run_prefix == '':
        run_prefix = os.path.splitext(os.path.basename(config_path))[0]

    load_model = config['load_model']
    pretrained_path = config['pretrained_path']  # path of pretrained model to load, if None, train from scratch
    adam_betas = config['adam_betas']
    adam_eps = config['adam_eps']
    scheduler_gamma = config['scheduler_gamma']
    eval_im_metrics = config['eval_im_metrics']
    cond_steps = config['cond_steps']  # conditional frames for the dynamics module during inference

    # model
    timestep_horizon = config['timestep_horizon']
    kp_range = config['kp_range']
    kp_activation = config['kp_activation']
    enc_channels = config['enc_channels']
    prior_channels = config['prior_channels']
    pad_mode = config['pad_mode']
    n_kp = config['n_kp']  # kp per patch in prior, best to leave at 1
    n_kp_prior = config['n_kp_prior']  # number of prior kp to filter for the kl
    n_kp_enc = config['n_kp_enc']  # total posterior kp
    patch_size = config['patch_size']  # prior patch size
    anchor_s = config['anchor_s']  # posterior patch/glimpse ratio of image size
    mu_scale_prior = config.get('mu_scale_prior', None)
    learned_feature_dim = config['learned_feature_dim']
    bg_learned_feature_dim = config.get('bg_learned_feature_dim', None)
    dropout = config['dropout']
    use_resblock = config['use_resblock']
    use_correlation_heatmaps = config['use_correlation_heatmaps']  # use heatmaps for tracking
    enable_enc_attn = config['enable_enc_attn']  # enable attention between patches in the particle encoder
    filtering_heuristic = config["filtering_heuristic"]  # filtering heuristic to filter prior keypoints
    use_actions = config.get("use_actions", False)  # use action-conditioned dynamics model
    max_beta_coef = config.get("max_beta_coef", 100)

    # optimization
    warmup_epoch = config['warmup_epoch']
    recon_loss_type = config['recon_loss_type']
    beta_kl = config['beta_kl']
    beta_dyn = config['beta_dyn']
    beta_rec = config['beta_rec']
    beta_dyn_rec = config['beta_dyn_rec']
    kl_balance = config['kl_balance']  # balance between visual features and the other particle attributes
    num_static_frames = config['num_static_frames']  # frames for which kl is calculated w.r.t constant prior params
    train_enc_prior = config['train_enc_prior']

    # priors
    sigma = config['sigma']  # std for constant kp prior, leave at 1 for deterministic chamfer-kl
    scale_std = config['scale_std']
    offset_std = config['offset_std']
    obj_on_alpha = config['obj_on_alpha']  # transparency beta distribution "a"
    obj_on_beta = config['obj_on_beta']  # transparency beta distribution "b"

    # transformer - PINT
    pint_layers = config['pint_layers']
    pint_heads = config['pint_heads']
    pint_dim = config['pint_dim']
    predict_delta = config['predict_delta']  # dynamics module predicts the delta from previous step
    start_epoch = config['start_dyn_epoch']

    model = ObjectDynamicsDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                              image_size=image_size, n_kp=n_kp, learned_feature_dim=learned_feature_dim,
                              pad_mode=pad_mode, sigma=sigma, bg_learned_feature_dim=bg_learned_feature_dim,
                              dropout=dropout, patch_size=patch_size, n_kp_enc=n_kp_enc,
                              n_kp_prior=n_kp_prior, kp_range=kp_range, kp_activation=kp_activation,
                              anchor_s=anchor_s, use_resblock=use_resblock,
                              timestep_horizon=timestep_horizon, predict_delta=predict_delta,
                              scale_std=scale_std, offset_std=offset_std, obj_on_alpha=obj_on_alpha,
                              obj_on_beta=obj_on_beta, pint_layers=pint_layers, pint_heads=pint_heads,
                              pint_dim=pint_dim, use_correlation_heatmaps=use_correlation_heatmaps,
                              enable_enc_attn=enable_enc_attn, filtering_heuristic=filtering_heuristic,
                              max_beta_coef=max_beta_coef, action_dim=action_dim, mu_scale_prior=mu_scale_prior)

    return model


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    pretrained_epoch = checkpoint['epoch'] + 1
    valid_loss = best_valid_loss = checkpoint['best_valid_loss']
    best_valid_epoch = checkpoint['best_valid_epoch']
    val_lpips = best_val_lpips = checkpoint['best_val_lpips']
    best_val_lpips_epoch = checkpoint['best_val_lpips_epoch']
    print(f"loaded model from checkpoint: {checkpoint_path}")

    return model
