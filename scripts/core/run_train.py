#!/usr/bin/env python

import json
import logging
import math
import time
from contextlib import nullcontext
from pprint import pformat
from pathlib import Path
from typing import Dict, Any
import yaml

import torch
from torch import Tensor, nn
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
import builtins
import os

from pathlib import Path
import PIL.ExifTags
import PIL.Image

if not hasattr(PIL.ExifTags, "Base"):
    class _PillowExifBase:
        Orientation = 274

    PIL.ExifTags.Base = _PillowExifBase
if not hasattr(PIL.Image, "ExifTags"):
    PIL.Image.ExifTags = PIL.ExifTags

import datetime as dt
import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.hub import HubMixin

TRAIN_CONFIG_NAME = "train_config.json"
DEFAULT_FORCE_XYZ_NAMES = (
    ("follower_ext_wrench_tcp_fx", "follower_ext_wrench_tcp_fy", "follower_ext_wrench_tcp_fz"),
    ("follower_ext_wrench_world_fx", "follower_ext_wrench_world_fy", "follower_ext_wrench_world_fz"),
    ("follower_ext_wrench_tcp_x", "follower_ext_wrench_tcp_y", "follower_ext_wrench_tcp_z"),
    ("follower_ext_wrench_world_x", "follower_ext_wrench_world_y", "follower_ext_wrench_world_z"),
    ("ext_wrench_tcp_fx", "ext_wrench_tcp_fy", "ext_wrench_tcp_fz"),
    ("ext_wrench_world_fx", "ext_wrench_world_fy", "ext_wrench_world_fz"),
    ("force_x", "force_y", "force_z"),
    ("fx", "fy", "fz"),
)


class FourierForceEncoder(nn.Module):
    """Encode current XYZ force as one ACT transformer token."""

    def __init__(
        self,
        force_dim: int,
        output_dim: int,
        fourier_dim: int = 8,
        min_period: float = 1e-3,
        max_period: float = 1.0,
    ):
        super().__init__()
        if fourier_dim % 2 != 0:
            raise ValueError(f"force_fourier_dim must be even, got {fourier_dim}")
        self.force_dim = force_dim
        self.fourier_dim = fourier_dim
        self.min_period = min_period
        self.max_period = max_period
        in_dim = force_dim + force_dim * fourier_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, force_xyz: Tensor) -> Tensor:
        if force_xyz.dim() != 2 or force_xyz.shape[-1] != self.force_dim:
            raise ValueError(f"Expected force tensor shape (B, {self.force_dim}), got {tuple(force_xyz.shape)}")
        force_xyz = force_xyz.to(dtype=torch.float32)
        fraction = torch.linspace(
            0.0,
            1.0,
            self.fourier_dim // 2,
            dtype=force_xyz.dtype,
            device=force_xyz.device,
        )
        period = self.min_period * (self.max_period / self.min_period) ** fraction
        scale = (2.0 * math.pi) / period
        sin_input = force_xyz.unsqueeze(-1) * scale
        fourier = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)
        fourier = fourier.reshape(force_xyz.shape[0], -1)
        return self.mlp(torch.cat([force_xyz, fourier], dim=-1))


def _find_force_xyz_indices(state_names: list[str], requested_names: list[str] | None = None) -> tuple[list[int], list[str]]:
    if requested_names:
        missing = [name for name in requested_names if name not in state_names]
        if missing:
            raise ValueError(
                f"Configured force_feature_names are missing from observation.state: {missing}. "
                f"Available names: {state_names}"
            )
        return [state_names.index(name) for name in requested_names], requested_names

    for candidate_names in DEFAULT_FORCE_XYZ_NAMES:
        if all(name in state_names for name in candidate_names):
            return [state_names.index(name) for name in candidate_names], list(candidate_names)

    raise ValueError(
        "Could not infer XYZ force columns from observation.state. "
        "Set policy.force_feature_names in train_cfg.yaml, for example "
        "['follower_ext_wrench_tcp_fx', 'follower_ext_wrench_tcp_fy', 'follower_ext_wrench_tcp_fz']. "
        f"Available observation.state names: {state_names}"
    )


def _install_force_xyz_act(policy: PreTrainedPolicy, ds_meta, cfg_policy) -> PreTrainedPolicy:
    from itertools import chain

    import einops
    import torchvision
    from torchvision.models._utils import IntermediateLayerGetter
    from torchvision.ops.misc import FrozenBatchNorm2d

    from lerobot.policies.act.modeling_act import (
        ACT,
        ACTEncoder,
        ACTDecoder,
        ACTSinusoidalPositionEmbedding2d,
        create_sinusoidal_pos_embedding,
    )
    from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

    state_feature = ds_meta.features.get(OBS_STATE)
    if state_feature is None or "names" not in state_feature:
        raise ValueError("act_force_xyz requires dataset meta feature 'observation.state' with named dimensions.")

    force_names = getattr(cfg_policy, "force_feature_names", None)
    force_indices, resolved_names = _find_force_xyz_indices(list(state_feature["names"]), force_names)
    force_fourier_dim = int(getattr(cfg_policy, "force_fourier_dim", 8))
    force_min_period = float(getattr(cfg_policy, "force_min_period", 1e-3))
    force_max_period = float(getattr(cfg_policy, "force_max_period", 1.0))

    class ForceXYZACT(ACT):
        def __init__(self, config):
            nn.Module.__init__(self)
            self.config = config
            self.force_indices = force_indices
            self.force_encoder = FourierForceEncoder(
                force_dim=3,
                output_dim=config.dim_model,
                fourier_dim=force_fourier_dim,
                min_period=force_min_period,
                max_period=force_max_period,
            )

            if self.config.use_vae:
                self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
                self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
                if self.config.robot_state_feature:
                    self.vae_encoder_robot_state_input_proj = nn.Linear(
                        self.config.robot_state_feature.shape[0], config.dim_model
                    )
                self.vae_encoder_action_input_proj = nn.Linear(
                    self.config.action_feature.shape[0],
                    config.dim_model,
                )
                self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
                num_input_token_encoder = 1 + config.chunk_size
                if self.config.robot_state_feature:
                    num_input_token_encoder += 1
                self.register_buffer(
                    "vae_encoder_pos_enc",
                    create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
                )

            if self.config.image_features:
                backbone_model = getattr(torchvision.models, config.vision_backbone)(
                    replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                    weights=config.pretrained_backbone_weights,
                    norm_layer=FrozenBatchNorm2d,
                )
                self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

            self.encoder = ACTEncoder(config)
            self.decoder = ACTDecoder(config)

            if self.config.robot_state_feature:
                self.encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            if self.config.env_state_feature:
                self.encoder_env_state_input_proj = nn.Linear(
                    self.config.env_state_feature.shape[0], config.dim_model
                )
            self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
            if self.config.image_features:
                self.encoder_img_feat_input_proj = nn.Conv2d(
                    backbone_model.fc.in_features, config.dim_model, kernel_size=1
                )

            n_1d_tokens = 2  # latent + force token
            if self.config.robot_state_feature:
                n_1d_tokens += 1
            if self.config.env_state_feature:
                n_1d_tokens += 1
            self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
            if self.config.image_features:
                self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

            self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
            self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])
            self._reset_parameters()

        def _reset_parameters(self):
            for p in chain(self.encoder.parameters(), self.decoder.parameters()):
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
            if self.config.use_vae and self.training:
                assert ACTION in batch, "actions must be provided when using the variational objective in training mode."

            batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

            if self.config.use_vae and ACTION in batch and self.training:
                cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
                if self.config.robot_state_feature:
                    robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
                action_embed = self.vae_encoder_action_input_proj(batch[ACTION])

                if self.config.robot_state_feature:
                    vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
                else:
                    vae_encoder_input = [cls_embed, action_embed]
                vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

                pos_embed = self.vae_encoder_pos_enc.clone().detach()
                cls_joint_is_pad = torch.full(
                    (batch_size, 2 if self.config.robot_state_feature else 1),
                    False,
                    device=batch[OBS_STATE].device,
                )
                key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)

                cls_token_out = self.vae_encoder(
                    vae_encoder_input.permute(1, 0, 2),
                    pos_embed=pos_embed.permute(1, 0, 2),
                    key_padding_mask=key_padding_mask,
                )[0]
                latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
                mu = latent_pdf_params[:, : self.config.latent_dim]
                log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
                latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
            else:
                mu = log_sigma_x2 = None
                latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                    batch[OBS_STATE].device
                )

            encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
            encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
            if self.config.robot_state_feature:
                encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
            force_xyz = batch[OBS_STATE][:, self.force_indices]
            encoder_in_tokens.append(self.force_encoder(force_xyz))
            if self.config.env_state_feature:
                encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

            if self.config.image_features:
                for img in batch[OBS_IMAGES]:
                    cam_features = self.backbone(img)["feature_map"]
                    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                    cam_features = self.encoder_img_feat_input_proj(cam_features)
                    cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                    cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                    encoder_in_tokens.extend(list(cam_features))
                    encoder_in_pos_embed.extend(list(cam_pos_embed))

            encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
            encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
            encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
            decoder_in = torch.zeros(
                (self.config.chunk_size, batch_size, self.config.dim_model),
                dtype=encoder_in_pos_embed.dtype,
                device=encoder_in_pos_embed.device,
            )
            decoder_out = self.decoder(
                decoder_in,
                encoder_out,
                encoder_pos_embed=encoder_in_pos_embed,
                decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
            )
            actions = self.action_head(decoder_out.transpose(0, 1))
            return actions, (mu, log_sigma_x2)

    policy.model = ForceXYZACT(policy.config).to(policy.config.device)
    policy.config.force_feature_names = resolved_names
    policy.config.force_feature_indices = force_indices
    policy.config.force_fourier_dim = force_fourier_dim
    policy.config.force_min_period = force_min_period
    policy.config.force_max_period = force_max_period
    logging.info(
        "Using act_force_xyz with force columns %s at observation.state indices %s",
        resolved_names,
        force_indices,
    )
    return policy


class TimedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that accumulates __getitem__ timing in the main process."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.reset_timing()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        start_time = time.perf_counter()
        item = self.dataset[index]
        elapsed = time.perf_counter() - start_time
        self.getitem_total_s += elapsed
        self.getitem_count += 1
        self.getitem_max_s = max(self.getitem_max_s, elapsed)
        return item

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def reset_timing(self):
        self.getitem_total_s = 0.0
        self.getitem_count = 0
        self.getitem_max_s = 0.0

    def pop_timing(self):
        timing = {
            "total_s": self.getitem_total_s,
            "count": self.getitem_count,
            "max_s": self.getitem_max_s,
        }
        self.reset_timing()
        return timing


class TrainPipelineConfig(HubMixin):
    def __init__(self, cfg: Dict[str, Any]):
        dataset = cfg["dataset"]
        # env = cfg["env"]
        policy = cfg["policy"]
        eval = cfg["eval"]
        wandb = cfg["wandb"]
    
        self.dataset: DatasetConfig = DatasetConfig(
            repo_id = dataset["repo_id"],
            root = dataset["root"]
            )

        # self.env: envs.EnvConfig | None = envs.EnvConfig(
        #     env_name = env["env_name"],
        #     env_type = env["env_type"],
        #     env_kwargs = env["env_kwargs"],
        # )
        self.env = None

        policy_type = policy["type"]
        if policy_type in {"act", "act_force_xyz"}:
            from lerobot.policies import ACTConfig
            self.policy = ACTConfig(
                device = policy["device"],
                repo_id = policy["repo_id"],
                push_to_hub = policy["push_to_hub"]
            )
            if policy_type == "act_force_xyz":
                self.policy.force_xyz_enabled = True
                self.policy.force_feature_names = policy.get("force_feature_names")
                self.policy.force_fourier_dim = policy.get("force_fourier_dim", 8)
                self.policy.force_min_period = policy.get("force_min_period", 1e-3)
                self.policy.force_max_period = policy.get("force_max_period", 1.0)
        elif policy_type == "diffusion":
            from lerobot.policies import DiffusionConfig
            self.policy = DiffusionConfig(
                device = policy["device"],
                repo_id = policy["repo_id"],
                push_to_hub = policy["push_to_hub"]
            )
        else:
            raise ValueError(f"no config for policy type: {policy_type}")

        # Set `dir` to where you would like to save all of the run outputs. If you run another training session
        # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
        self.output_dir: Path | None = Path(cfg["output_dir"]) if cfg["output_dir"] else None
        self.job_name: str | None = cfg["job_name"]
        # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
        # `dir` is the directory of an existing run with at least one checkpoint in it.
        # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
        # regardless of what's provided with the training command at the time of resumption.
        self.resume: bool = cfg["resume"]
        # `seed` is used for training (eg: model initialization, dataset shuffling)
        # AND for the evaluation environments.
        self.seed: int | None = cfg["seed"]
        # Number of workers for the dataloader.
        self.num_workers: int = cfg["num_workers"]
        self.batch_size: int = cfg["batch_size"]
        self.steps: int = cfg["steps"]
        self.eval_freq: int = cfg["eval_freq"]
        self.log_freq: int = cfg["log_freq"]
        self.save_checkpoint: bool = cfg["save_checkpoint"]
        self.save_freq: int = cfg["save_freq"]
        self.use_policy_training_preset: bool = cfg["use_policy_training_preset"]
        
        self.eval: EvalConfig = EvalConfig(
            n_episodes = eval["n_episodes"],
            batch_size = eval["batch_size"]
        )

        self.wandb: WandBConfig = WandBConfig(
            enable = wandb["enable"],
            disable_artifact = wandb.get("disable_artifact", False),
            project = wandb["project"],
            entity = wandb.get("entity"),  # 使用 get 方法，允许键不存在
            notes = wandb.get("notes"),    # 使用 get 方法，允许键不存在
            run_id = wandb.get("run_id"),  # 使用 get 方法，允许键不存在
            mode = wandb["mode"]
        )

    def __post_init__(self):
        self.checkpoint_path = None

    def validate(self):

        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Only load the policy config
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        elif self.resume:
            # The entire train config is already loaded, we just need to get the checkpoint dir
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )
            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )
            policy_path = Path(config_path).parent
            self.policy.pretrained_path = policy_path
            self.checkpoint_path = policy_path.parent

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.resume and self.output_dir:
            now = dt.datetime.now()
            self.output_dir = self.output_dir / f"{now:%Y%m%d_%H%M%S_%f}"
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

        if self.policy.push_to_hub and not self.policy.repo_id:
            raise ValueError(
                "'policy.repo_id' argument missing. Please specify it to push the model to the hub."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict:
        """将配置对象转换为可序列化的字典"""
        result = {}
        for key, value in self.__dict__.items():
            # 跳过私有属性
            if key.startswith('_'):
                continue
                
            # 处理特殊类型的属性
            if key == "policy":
                result[key] = self._serialize_policy_config(value)
            elif value is None:
                result[key] = None
            elif isinstance(value, (str, int, float, bool)):
                result[key] = value
            elif isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, (list, tuple)):
                result[key] = [self._serialize_item(item) for item in value]
            elif isinstance(value, dict):
                result[key] = {k: self._serialize_item(v) for k, v in value.items()}
            elif hasattr(value, 'to_dict'):
                # 如果属性有 to_dict 方法，调用它
                result[key] = value.to_dict()
            else:
                # 对于其他对象，使用安全的序列化方法
                result[key] = self._serialize_item(value)
        return result

    def _serialize_item(self, item):
        """安全地序列化单个项目"""
        if item is None:
            return None
        elif isinstance(item, (str, int, float, bool)):
            return item
        elif isinstance(item, Path):
            return str(item)
        elif isinstance(item, (list, tuple)):
            return [self._serialize_item(i) for i in item]
        elif isinstance(item, dict):
            return {k: self._serialize_item(v) for k, v in item.items()}
        elif hasattr(item, 'to_dict'):
            return item.to_dict()
        elif hasattr(item, '__dict__'):
            # 对于复杂对象，只序列化基本属性
            return self._serialize_simple_object(item)
        else:
            # 最后手段：返回字符串表示
            return str(item)
    
    def _serialize_simple_object(self, obj):
        """安全地序列化简单对象，避免循环引用"""
        result = {}
        for attr_name, attr_value in obj.__dict__.items():
            # 跳过私有属性和复杂对象
            if attr_name.startswith('_'):
                continue
            
            try:
                # 只序列化基本类型
                if isinstance(attr_value, (str, int, float, bool, type(None))):
                    result[attr_name] = attr_value
                elif isinstance(attr_value, Path):
                    result[attr_name] = str(attr_value)
            except:
                # 如果序列化失败，跳过该属性
                continue
        
        # 如果没有任何可序列化的属性，返回类型名称
        if not result:
            return f"<{obj.__class__.__name__}>"
        
        return result

    def _serialize_policy_config(self, policy_cfg):
        data = self._serialize_simple_object(policy_cfg)
        policy_type = getattr(policy_cfg, "type", None)
        if getattr(policy_cfg, "force_xyz_enabled", False):
            policy_type = "act_force_xyz"
        if policy_type is not None:
            data["type"] = policy_type
        return data

    def _save_pretrained(self, save_directory: Path) -> None:
        # 使用手动实现的 to_dict 方法保存配置
        config_dict = self.to_dict()
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)



def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    forward_start = time.perf_counter()
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    train_metrics.forward_s = time.perf_counter() - forward_start

    # Use accelerator's backward method
    backward_start = time.perf_counter()
    accelerator.backward(loss)
    train_metrics.backward_s = time.perf_counter() - backward_start

    # Clip gradients if specified
    grad_clip_start = time.perf_counter()
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )
    train_metrics.grad_clip_s = time.perf_counter() - grad_clip_start

    # Optimizer step
    optimizer_start = time.perf_counter()
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()
    train_metrics.optimizer_s = time.perf_counter() - optimizer_start

    # Step through pytorch scheduler at every batch instead of epoch
    scheduler_start = time.perf_counter()
    if lr_scheduler is not None:
        lr_scheduler.step()
    train_metrics.scheduler_s = time.perf_counter() - scheduler_start

    # Update internal buffers if policy has update method
    policy_update_start = time.perf_counter()
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
    train_metrics.policy_update_s = time.perf_counter() - policy_update_start

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def run_train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if is_main_process:
            logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    if getattr(cfg.policy, "force_xyz_enabled", False):
        pretrained_path = cfg.policy.pretrained_path
        cfg.policy.pretrained_path = None
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
        )
        policy = _install_force_xyz_act(policy, dataset.meta, cfg.policy)
        cfg.policy.pretrained_path = pretrained_path
        if pretrained_path is not None:
            from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
            from safetensors.torch import load_model as load_model_as_safetensor

            model_file = Path(pretrained_path) / SAFETENSORS_SINGLE_FILE
            if not model_file.exists():
                raise FileNotFoundError(f"Expected policy weights at {model_file}")
            missing_keys, unexpected_keys = load_model_as_safetensor(policy, str(model_file), strict=False)
            if is_main_process:
                logging.info(
                    "Loaded act_force_xyz weights from %s (missing=%s, unexpected=%s)",
                    model_file,
                    list(missing_keys),
                    list(unexpected_keys),
                )
            policy.to(cfg.policy.device)
    else:
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
        )

    print("=== 策略输入特征 (状态量) ===")
    for key, value in policy.config.input_features.items():
        print(f'{key}: {value}')

    print("\n=== 策略输出特征 (动作) ===")
    for key, value in policy.config.output_features.items():
        print(f'{key}: {value}')

    print("\n=== 策略配置 ===")
    print(f"类型: {policy.config.type}")
    print(f"设备: {policy.config.device}")

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader_dataset = TimedDataset(dataset) if cfg.num_workers == 0 else dataset
    if is_main_process:
        if cfg.num_workers == 0:
            logging.info(
                "Step timing enabled: data_s=dl_next_s+prep_s, "
                "dl_next_s=DataLoader next(batch), prep_s=preprocessor(batch), "
                "getitem_s=sum(dataset.__getitem__), getitem_max_s=slowest sample in batch"
            )
        else:
            logging.info(
                "Step timing enabled: data_s=dl_next_s+prep_s, "
                "dl_next_s=DataLoader next(batch), prep_s=preprocessor(batch). "
                "getitem_s is unavailable when num_workers > 0."
            )

    dataloader = torch.utils.data.DataLoader(
        dataloader_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "dataloader_next_s": AverageMeter("dl_next_s", ":.3f"),
        "preprocess_s": AverageMeter("prep_s", ":.3f"),
        "dataset_getitem_s": AverageMeter("getitem_s", ":.3f"),
        "dataset_getitem_max_s": AverageMeter("getitem_max_s", ":.3f"),
        "dataset_getitem_count": AverageMeter("getitem_n", ":.0f"),
        "forward_s": AverageMeter("fwd_s", ":.3f"),
        "backward_s": AverageMeter("bwd_s", ":.3f"),
        "grad_clip_s": AverageMeter("clip_s", ":.3f"),
        "optimizer_s": AverageMeter("optim_s", ":.3f"),
        "scheduler_s": AverageMeter("sched_s", ":.3f"),
        "policy_update_s": AverageMeter("polupd_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()

        dataloader_start = time.perf_counter()
        batch = next(dl_iter)
        dataloader_next_s = time.perf_counter() - dataloader_start

        if isinstance(dataloader_dataset, TimedDataset):
            dataset_timing = dataloader_dataset.pop_timing()
        else:
            dataset_timing = {"total_s": 0.0, "count": 0, "max_s": 0.0}

        preprocess_start = time.perf_counter()
        batch = preprocessor(batch)
        preprocess_s = time.perf_counter() - preprocess_start

        train_tracker.dataloading_s = time.perf_counter() - start_time
        train_tracker.dataloader_next_s = dataloader_next_s
        train_tracker.preprocess_s = preprocess_s
        train_tracker.dataset_getitem_s = dataset_timing["total_s"]
        train_tracker.dataset_getitem_max_s = dataset_timing["max_s"]
        train_tracker.dataset_getitem_count = dataset_timing["count"]

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("âˆ‘rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "train_cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    train_cfg = TrainPipelineConfig(cfg["train"])

    run_train(train_cfg)


if __name__ == "__main__":
    main()
