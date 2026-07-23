from abc import abstractmethod
from typing import List

class DefaultTrainingConfig:
    """Default training configuration. """

    agent: str = "drq"
    max_traj_length: int = 100
    batch_size: int = 128
    cta_ratio: int = 2
    discount: float = 0.97

    max_steps: int = 1000000
    replay_buffer_capacity: int = 200000

    random_steps: int = 0
    training_starts: int = 100
    steps_per_update: int = 50

    log_period: int = 10
    eval_period: int = 2000

    # "resnet" for ResNet10 from scratch and "resnet-pretrained" for frozen ResNet10 with pretrained weights
    encoder_type: str = "resnet-pretrained"
    demo_path: str = None
    checkpoint_period: int = 0
    buffer_period: int = 0

    eval_checkpoint_step: int = 0
    eval_n_trajs: int = 5

    image_keys: List[str] = None
    classifier_keys: List[str] = None
    proprio_keys: List[str] = None
    
    # "single-arm-learned-gripper", "dual-arm-learned-gripper" for with learned gripper, 
    # "single-arm-fixed-gripper", "dual-arm-fixed-gripper" for without learned gripper (i.e. pregrasped)
    setup_mode: str = "single-arm-fixed-gripper"

    # Causal masking in the actor's fused latent space.
    causal_mask_enabled: bool = True
    # Apply mask directly to sampled batch latents instead of appending extra data.
    causal_mask_inplace: bool = True
    # In inplace mode, choose which sampled real-data sources receive the mask.
    causal_mask_online_batch: bool = True
    causal_mask_demo_batch: bool = True
    # Threshold masks remain logged for diagnostics; augmentation uses the lowest-CMI ratio.
    causal_mask_threshold: float = 0.3
    # Fit the causal model from collected demos before actor communication starts.
    causal_pretrain_steps: int = 5000
    # Continue fitting from demos plus synced intervention transitions during RL.
    causal_model_update_interval: int = 0
    causal_model_train_steps: int = 0
    # Save/load the causal model separately from the policy checkpoint.
    # The actual path is <checkpoint_path>/<causal_model_checkpoint_subdir>.
    causal_model_checkpoint_enabled: bool = True
    causal_model_checkpoint_subdir: str = "causal_model"
    # When a causal checkpoint exists, load it before policy learning and skip
    # repeated demo pretraining by default.
    causal_model_skip_pretrain_if_loaded: bool = True
    # Online causal model updates use old demos plus successful
    # human-intervention transitions with these target sampling ratios.
    causal_model_online_demo_ratio: float = 0.8
    causal_model_online_success_intervention_ratio: float = 0.2
    causal_action_samples: int = 64

    # Fraction of paired online/demo rows to mask.
    causal_mask_ratio: float = 1.0
    # Mask mode uses the paired demo batch mean as the neutral latent baseline.
    causal_mask_baseline: str = "mean"
    # 1.0 is a hard mask; smaller values retain part of the selected features.
    causal_mask_strength: float = 0.3
    # This ratio controls selected dimensions.
    causal_mask_max_ratio: float = 0.3
    causal_mask_visual_only: bool = True
    causal_mask_proprio_latent_dim: int = 64
    @abstractmethod
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        raise NotImplementedError
    
    @abstractmethod
    def process_demos(self, demo):
        raise NotImplementedError
    
