import json
import traceback
from copy import deepcopy
import warnings

import gym

from envs.wrappers.features import FeaturesWrapper
from ocr.akorn.source.models.savi.model import AkornSAVi

try:
    import isaacgym
except ImportError:
    print('isaacgym is not installed')

import torch

from envs.wrappers.collect_episodes_wrapper import CollectEpisodes
from envs.wrappers.ddlp import DynamicDDLPExtractorWrapper, StaticDDLPExtractorWrapper
from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper


def missing_dependencies(task):
    raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')


try:
    from envs.dmcontrol import make_env as make_dm_control_env
except:
    make_dm_control_env = missing_dependencies
try:
    from envs.maniskill import make_env as make_maniskill_env
except:
    make_maniskill_env = missing_dependencies
try:
    from envs.metaworld import make_env as make_metaworld_env
except:
    make_metaworld_env = missing_dependencies
try:
    from envs.myosuite import make_env as make_myosuite_env
except:
    make_myosuite_env = missing_dependencies
try:
    from envs.maniskill3 import make_env as make_maniskill3_env
except:
    make_maniskill3_env = missing_dependencies
try:
    from envs.robosuite_env import make_env as make_robosuite_env
except:
    make_robosuite_env = missing_dependencies
try:
    from envs.isaac.isaac_env_wrappers import make_env as make_isaac_env
except:
    make_isaac_env = missing_dependencies
try:
    from envs.cw_envs.target import make_env as make_cw_env
except:
    make_cw_env = missing_dependencies
try:
    from envs.mof import make_env as make_mof_env
except:
    make_mof_env = missing_dependencies

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
    """
    Make a multi-task environment for TD-MPC2 experiments.
    """
    print('Creating multi-task environment with tasks:', cfg.tasks)
    envs = []
    for task in cfg.tasks:
        _cfg = deepcopy(cfg)
        _cfg.task = task
        _cfg.multitask = False
        env = make_env(_cfg)
        if env is None:
            raise ValueError('Unknown task:', task)
        envs.append(env)
    env = MultitaskWrapper(cfg, envs)
    cfg.obs_shapes = env._obs_dims
    cfg.action_dims = env._action_dims
    cfg.episode_lengths = env._episode_lengths
    return env


def make_env(cfg, **kwargs):
    """
    Make an environment for TD-MPC2 experiments.
    """
    gym.logger.set_level(40)
    if cfg.multitask:
        env = make_multitask_env(cfg)

    else:
        env = None
        for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env, make_maniskill3_env,
                   make_robosuite_env, make_isaac_env, make_cw_env, make_mof_env]:
            try:
                env = fn(cfg)
            except ValueError:
                print(traceback.format_exc())
        if env is None:
            raise ValueError(
                f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')

    if cfg.do_collect_episodes:
        env = CollectEpisodes(env, cfg)

    obs_type = cfg.get('obs', 'state')
    if obs_type == 'rgb':
        env = PixelWrapper(cfg, env, num_frames=cfg.num_frames, render_size=cfg.obs_size)
    elif obs_type == 'slots':
        from envs.wrappers.slots import SlotExtractorWrapper

        slot_extractor_model = cfg['slot_extractor_model']
        if slot_extractor_model == 'dinosaur':
            from ocr.tools import SlotExtractor, Dinosaur

            sa_model = Dinosaur(cfg.dino_model_name, cfg.n_slots, cfg.slot_dim, cfg.input_feature_dim, cfg.num_patches,
                                cfg.features)
            state_dict = torch.load(cfg.slot_extractor_checkpoint_path)['state_dict']
            state_dict = {key[len('models.'):]: value for key, value in state_dict.items()}
            sa_model.load_state_dict(state_dict)
        elif slot_extractor_model == 'akornsaur':
            from ocr.tools import SlotExtractor
            from ema_pytorch import EMA
            from ocr.akorn.source.models.objs.knet import AKOrN
            from ocr.akorn.source.models.slot_attention.akornsaur import AkornSAur
            from ocr.akorn.source.models.slot_attention.decoders import MLPDecoder
            from ocr.akorn.source.models.slot_attention.initializers import RandomInit
            from ocr.akorn.source.models.slot_attention.networks import MLP
            from ocr.akorn.source.models.slot_attention.slot_attention import SlotAttention

            n_patches = (cfg.obs_size // cfg.psize) ** 2
            encoder = AKOrN(
                cfg.N,
                ch=cfg.ch,
                L=cfg.L,
                T=cfg.T,
                J=cfg.J,
                use_omega=cfg.use_omega,
                global_omg=cfg.global_omg,
                c_norm=cfg.c_norm,
                psize=cfg.psize,
                imsize=cfg.obs_size,
                autorescale=cfg.autorescale,
                maxpool=cfg.maxpool,
                project=cfg.project,
                heads=cfg.heads,
                use_ro_x=cfg.use_ro_x,
                no_ro=cfg.no_ro,
                gta=cfg.gta,
            )

            encoder = EMA(encoder)
            encoder = encoder.ema_model

            features_projector = MLP(
                inp_dim=cfg.ch,
                outp_dim=cfg.slot_dim,
                hidden_dims=[2 * 256],
                initial_layer_norm=True, )

            initializer = RandomInit(
                n_slots=cfg.n_slots,
                dim=cfg.slot_dim,
                per_slot_initialization=cfg.per_slot_initialization,
                deterministic_initialization=cfg.deterministic_initialization
            )

            slot_attention = SlotAttention(
                inp_dim=cfg.slot_dim,
                slot_dim=cfg.slot_dim,
                n_iters=3,
                use_mlp=True, )

            decoder = MLPDecoder(inp_dim=cfg.slot_dim, outp_dim=cfg.ch, hidden_dims=[512, 512, 512], n_patches=n_patches)
            sa_model = AkornSAur(encoder, features_projector, initializer, slot_attention, decoder,
                                  is_encoder_frozen=True)
            weights = torch.load(cfg.slot_extractor_checkpoint_path, weights_only=True)['model']
            sa_model.load_state_dict(weights)
        elif slot_extractor_model == 'ksavi':
            from envs.wrappers.savi_wrapper import SlotExtractor
            from ema_pytorch import EMA
            from ocr.akorn.source.models.objs.knet import AKOrN
            from ocr.akorn.source.models.slot_attention.decoders import MLPDecoder
            from ocr.akorn.source.models.savi.initializer import Learned
            from ocr.akorn.source.models.slot_attention.networks import MLP
            from ocr.akorn.source.models.slot_attention.slot_attention import SlotAttention
            from ocr.akorn.source.models.savi.predictor import TransformerPredictor
            from ocr.akorn.source.models.savi import Corrector

            n_patches = (cfg.obs_size // cfg.psize) ** 2
            encoder = AKOrN(
                cfg.N,
                ch=cfg.ch,
                L=cfg.L,
                T=cfg.T,
                J=cfg.J,
                use_omega=cfg.use_omega,
                global_omg=cfg.global_omg,
                c_norm=cfg.c_norm,
                psize=cfg.psize,
                imsize=cfg.obs_size,
                autorescale=cfg.autorescale,
                maxpool=cfg.maxpool,
                project=cfg.project,
                heads=cfg.heads,
                use_ro_x=cfg.use_ro_x,
                no_ro=cfg.no_ro,
                gta=cfg.gta,
            )

            encoder = EMA(encoder)
            encoder = encoder.ema_model

            features_projector = MLP(
                inp_dim=256,
                outp_dim=cfg.slot_dim,
                hidden_dims=[2 * 256],
                initial_layer_norm=True, )

            initializer = Learned(num_slots=cfg.n_slots, slot_dim=cfg.slot_dim)

            slot_attention = Corrector(
                num_slots=cfg.n_slots,
                slot_dim=cfg.slot_dim,
                feature_dim=cfg.slot_dim,
                num_iterations=1,
                num_initial_iterations=3,
                hidden_dim=4 * cfg.slot_dim,
            )

            decoder = MLPDecoder(inp_dim=cfg.slot_dim, outp_dim=256, hidden_dims=[512, 512, 512], n_patches=n_patches)
            predictor = TransformerPredictor(slot_dim=cfg.slot_dim, action_dim=-1,)
            sa_model = AkornSAVi(encoder, features_projector, initializer, slot_attention, decoder, predictor,
                                  is_encoder_frozen=True)
            weights = torch.load(cfg.slot_extractor_checkpoint_path, weights_only=True)['model']
            weights = {key: value for key, value in weights.items() if not key.startswith('image_decoder')}
            sa_model.load_state_dict(weights)
        elif slot_extractor_model == 'slot-contrast':
            from envs.wrappers.savi_wrapper import SlotExtractor
            from ema_pytorch import EMA
            from ocr.akorn.source.models.objs.knet import AKOrN
            from ocr.akorn.source.models.slotcontrast.modules.networks import MLP, TransformerEncoder
            from ocr.akorn.source.models.slotcontrast.modules.initializer import FixedLearnedInit
            from ocr.akorn.source.models.slotcontrast.modules.groupers import SlotAttention
            from ocr.akorn.source.models.slotcontrast.modules.decoders import MLPDecoder
            from ocr.akorn.source.models.slotcontrast.modules.video import LatentProcessor
            from ocr.akorn.source.models.slotcontrast.model import SlotContrastAkornSAur

            n_patches = (cfg.obs_size // cfg.psize) ** 2
            encoder = AKOrN(
                cfg.N,
                ch=cfg.ch,
                L=cfg.L,
                T=cfg.T,
                J=cfg.J,
                use_omega=cfg.use_omega,
                global_omg=cfg.global_omg,
                c_norm=cfg.c_norm,
                psize=cfg.psize,
                imsize=cfg.obs_size,
                autorescale=cfg.autorescale,
                maxpool=cfg.maxpool,
                project=cfg.project,
                heads=cfg.heads,
                use_ro_x=cfg.use_ro_x,
                no_ro=cfg.no_ro,
                gta=cfg.gta,
            )

            encoder = EMA(encoder)
            encoder = encoder.ema_model

            ch = 256

            encoder_output_transform = MLP(
                inp_dim=ch, outp_dim=cfg.slot_dim, hidden_dims=[2 * ch], initial_layer_norm=True,
            )
            initializer = FixedLearnedInit(n_slots=cfg.n_slots, dim=cfg.slot_dim, normalize_slots=cfg.normalize_slots)
            slot_attention = SlotAttention(
                inp_dim=cfg.slot_dim, slot_dim=cfg.slot_dim, n_iters=2, use_mlp=True,
                normalize_slots=cfg.normalize_slots
            )
            decoder = MLPDecoder(inp_dim=cfg.slot_dim, outp_dim=ch, hidden_dims=[1024, 1024, 1024],
                                 n_patches=n_patches)
            predictor = TransformerEncoder(dim=cfg.slot_dim, n_blocks=1, n_heads=4,
                                           normalize_output=cfg.normalize_slots)
            latent_processor = LatentProcessor(corrector=slot_attention, predictor=predictor,
                                               first_step_corrector_args={'n_iters': 3})
            sa_model = SlotContrastAkornSAur(
                encoder=encoder, encoder_output_transform=encoder_output_transform, initializer=initializer,
                decoder=decoder, latent_processor=latent_processor, is_encoder_frozen=True,
            )
            weights = torch.load(cfg.slot_extractor_checkpoint_path, weights_only=True)['model']
            sa_model.load_state_dict(weights)
        elif slot_extractor_model == 'savi':
            from sold.modeling.savi import Corrector, FullyConvolutionalEncoder, FullyConvolutionalDecoder, TransformerPredictor, Learned
            from envs.wrappers.savi_wrapper import SlotExtractor, load_savi_module
            from sold.modeling.savi.model import SAVi

            assert cfg.pre_initialize_slots, f'With SAVi encoder must use pre_initialized_slots == True'
            corrector = Corrector(cfg.n_slots, cfg.slot_dim, feature_dim=cfg.savi_feature_dim,
                                  num_iterations=cfg.savi_num_iterations,
                                  num_initial_iterations=cfg.savi_num_initial_iterations,
                                  hidden_dim=cfg.savi_hidden_dim,)
            predictor = TransformerPredictor(cfg.slot_dim, action_dim=-1,)
            encoder = FullyConvolutionalEncoder(image_size=env.observation_space.shape[:-1],
                                                num_channels=cfg.savi_num_channels, kernel_size=cfg.savi_kernel_size,
                                                feature_dim=cfg.savi_feature_dim,)
            decoder = FullyConvolutionalDecoder(image_size=env.observation_space.shape[:-1], slot_dim=cfg.slot_dim,
                                                in_channels=cfg.slot_dim, num_channels=cfg.savi_num_channels,
                                                kernel_size=cfg.savi_kernel_size,)
            slot_initializer = Learned(cfg.n_slots, cfg.slot_dim,)
            sa_model = SAVi(corrector, predictor, encoder, decoder, slot_initializer,)
            load_savi_module(sa_model, cfg.slot_extractor_checkpoint_path,)
        else:
            raise ValueError(f'Unexpected slot_extractor_model={slot_extractor_model}')

        sa_model = sa_model.requires_grad_(False)
        sa_model = sa_model.eval()
        slot_extractor = SlotExtractor(model=sa_model, device=cfg.slot_extractor_device)
        env = SlotExtractorWrapper(cfg, env, slot_extractor)
    elif obs_type == 'ddlp':
        ddlp = kwargs['extractor']
        assert ddlp.action_dim == env.action_space.shape[0]
        config_path = cfg.ddlp_config_path
        with open(config_path, 'r') as file_obj:
            config = json.load(file_obj)

        if cfg.transition_model_type == 'ddlp':
            env = DynamicDDLPExtractorWrapper(env, ddlp, device=cfg.device, num_static_frames=config['num_static_frames'],
                                              train_enc_prior=config['train_enc_prior'],
                                              save_folder=kwargs.get('save_folder', None))
        elif cfg.transition_model_type == 'gnn':
            env = StaticDDLPExtractorWrapper(env, ddlp, device=cfg.device, num_static_frames=config['num_static_frames'],
                                             train_enc_prior=config['train_enc_prior'], num_frames=cfg.num_frames)
        else:
            assert False
    elif obs_type == 'state' and cfg['slot_extractor_model'] == 'akornsaur':
        from ocr.tools import SlotExtractor
        from ema_pytorch import EMA
        from ocr.akorn.source.models.objs.knet import AKOrN
        from ocr.akorn.source.models.slot_attention.akornsaur import AkornSAur
        from ocr.akorn.source.models.slot_attention.decoders import MLPDecoder
        from ocr.akorn.source.models.slot_attention.initializers import RandomInit
        from ocr.akorn.source.models.slot_attention.networks import MLP
        from ocr.akorn.source.models.slot_attention.slot_attention import SlotAttention

        n_patches = (cfg.obs_size // cfg.psize) ** 2
        encoder = AKOrN(
            cfg.N,
            ch=cfg.ch,
            L=cfg.L,
            T=cfg.T,
            J=cfg.J,
            use_omega=cfg.use_omega,
            global_omg=cfg.global_omg,
            c_norm=cfg.c_norm,
            psize=cfg.psize,
            imsize=cfg.obs_size,
            autorescale=cfg.autorescale,
            maxpool=cfg.maxpool,
            project=cfg.project,
            heads=cfg.heads,
            use_ro_x=cfg.use_ro_x,
            no_ro=cfg.no_ro,
            gta=cfg.gta,
        )

        encoder = EMA(encoder)
        encoder = encoder.ema_model

        features_projector = MLP(
            inp_dim=cfg.ch,
            outp_dim=cfg.slot_dim,
            hidden_dims=[2 * 256],
            initial_layer_norm=True, )

        initializer = RandomInit(
            n_slots=cfg.n_slots,
            dim=cfg.slot_dim,
            per_slot_initialization=cfg.per_slot_initialization,
            deterministic_initialization=cfg.deterministic_initialization
        )

        slot_attention = SlotAttention(
            inp_dim=cfg.slot_dim,
            slot_dim=cfg.slot_dim,
            n_iters=3,
            use_mlp=True, )

        decoder = MLPDecoder(inp_dim=cfg.slot_dim, outp_dim=cfg.ch, hidden_dims=[512, 512, 512], n_patches=n_patches)
        sa_model = AkornSAur(encoder, features_projector, initializer, slot_attention, decoder,
                              is_encoder_frozen=True)
        weights = torch.load(cfg.slot_extractor_checkpoint_path, weights_only=True)['model']
        sa_model.load_state_dict(weights)

        features_extractor = sa_model.encoder.to(cfg.slot_extractor_device)
        env = FeaturesWrapper(cfg, env, features_extractor)

    if not cfg.multitask:
        env = TensorWrapper(env)

    try:  # Dict
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except:  # Box
        cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.action_lower_bound = env.action_space.low.tolist()
    cfg.action_upper_bound = env.action_space.high.tolist()
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = cfg.get('seed_steps', max(1000, 5 * cfg.episode_length))
    return env
