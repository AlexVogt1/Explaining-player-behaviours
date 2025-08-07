"""List of md_env."""
from logging import config
from typing import Final

from gym_md.envs.md_env import MdEnvBase
from gym_md.envs.md_policy_env import MdPolicyEnv

default_config= {
    "action_type": "policy",
    "action_space_type": "discrete",
    "obs_type": "grid",
    "base_path":  "./play_style_models/base/", 
}

class TestMdEnv(MdEnvBase):
    """TestMdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "test"
        super(TestMdEnv, self).__init__(stage_name=stage_name)


class EdgeMdEnv(MdEnvBase):
    """EdgeMdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "edge"
        super(EdgeMdEnv, self).__init__(stage_name=stage_name)


class HardMdEnv(MdEnvBase):
    """HardMdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "hard"
        super(HardMdEnv, self).__init__(stage_name=stage_name)


class Random1MdEnv(MdEnvBase):
    """Random1Env Class."""

    def __init__(self):
        stage_name: Final[str] = "random_1"
        super(Random1MdEnv, self).__init__(stage_name=stage_name)


class Random2MdEnv(MdEnvBase):
    """Random1Env Class."""

    def __init__(self):
        stage_name: Final[str] = "random_2"
        super(Random2MdEnv, self).__init__(stage_name=stage_name)


class Gene1MdEnv(MdEnvBase):
    """Random1Env Class."""

    def __init__(self):
        stage_name: Final[str] = "gene_1"
        super(Gene1MdEnv, self).__init__(stage_name=stage_name)


class Gene2MdEnv(MdEnvBase):
    """Random1Env Class."""

    def __init__(self):
        stage_name: Final[str] = "gene_2"
        super(Gene2MdEnv, self).__init__(stage_name=stage_name)


class Strand1MdEnv(MdEnvBase):
    """Strand1MdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "strand_1"
        super(Strand1MdEnv, self).__init__(stage_name=stage_name)


class Strand2MdEnv(MdEnvBase):
    """Strand2MdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "strand_2"
        super(Strand2MdEnv, self).__init__(stage_name=stage_name)


class Strand3MdEnv(MdEnvBase):
    """Strand3MdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "strand_3"
        super(Strand3MdEnv, self).__init__(stage_name=stage_name)


class Strand4MdEnv(MdEnvBase):
    """Strand4MdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "strand_4"
        super(Strand4MdEnv, self).__init__(stage_name=stage_name)


class Strand5MdEnv(MdEnvBase):
    """Strand5MdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "strand_5"
        super(Strand5MdEnv, self).__init__(stage_name=stage_name)


class Check1MdEnv(MdEnvBase):
    """Check1MdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "check_1"
        super(Check1MdEnv, self).__init__(stage_name=stage_name)


class Check2MdEnv(MdEnvBase):
    """Check2MdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "check_2"
        super(Check2MdEnv, self).__init__(stage_name=stage_name)


class Check3MdEnv(MdEnvBase):
    """Check3MdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "check_3"
        super(Check3MdEnv, self).__init__(stage_name=stage_name)


class Holmgard0MdEnv(MdEnvBase):
    """Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "holmgard_0"
        super(Holmgard0MdEnv, self).__init__(stage_name=stage_name)


class Holmgard1MdEnv(MdEnvBase):
    """Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "holmgard_1"
        super(Holmgard1MdEnv, self).__init__(stage_name=stage_name)


class Holmgard2MdEnv(MdEnvBase):
    """Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "holmgard_2"
        super(Holmgard2MdEnv, self).__init__(stage_name=stage_name)


class Holmgard3MdEnv(MdEnvBase):
    """Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "holmgard_3"
        super(Holmgard3MdEnv, self).__init__(stage_name=stage_name)


class Holmgard4MdEnv(MdEnvBase):
    """Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "holmgard_4"
        super(Holmgard4MdEnv, self).__init__(stage_name=stage_name)


class Holmgard5MdEnv(MdEnvBase):
    """Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "holmgard_5"
        super(Holmgard5MdEnv, self).__init__(stage_name=stage_name)


class Holmgard6MdEnv(MdEnvBase):
    """Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "holmgard_6"
        super(Holmgard6MdEnv, self).__init__(stage_name=stage_name)


class Holmgard7MdEnv(MdEnvBase):
    """Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "holmgard_7"
        super(Holmgard7MdEnv, self).__init__(stage_name=stage_name)


class Holmgard8MdEnv(MdEnvBase):
    """Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "holmgard_8"
        super(Holmgard8MdEnv, self).__init__(stage_name=stage_name)


class Holmgard9MdEnv(MdEnvBase):
    """Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "holmgard_9"
        super(Holmgard9MdEnv, self).__init__(stage_name=stage_name)


class Holmgard10MdEnv(MdEnvBase):
    """Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "holmgard_10"
        super(Holmgard10MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgard0MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgard_0"
        super(ConstantHolmgard0MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgard1MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgard_1"
        super(ConstantHolmgard1MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgard2MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgard_2"
        super(ConstantHolmgard2MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgard3MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgard_3"
        super(ConstantHolmgard3MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgard4MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgard_4"
        super(ConstantHolmgard4MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgard5MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgard_5"
        super(ConstantHolmgard5MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgard6MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgard_6"
        super(ConstantHolmgard6MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgard7MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgard_7"
        super(ConstantHolmgard7MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgard8MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgard_8"
        super(ConstantHolmgard8MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgard9MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgard_9"
        super(ConstantHolmgard9MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgard10MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgard_10"
        super(ConstantHolmgard10MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgardLarge0MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgardLarge_0"
        super(ConstantHolmgardLarge0MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgardLarge1MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgardLarge_1"
        super(ConstantHolmgardLarge1MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgardLarge2MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgardLarge_2"
        super(ConstantHolmgardLarge2MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgardLarge3MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgardLarge_3"
        super(ConstantHolmgardLarge3MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgardLarge4MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgardLarge_4"
        super(ConstantHolmgardLarge4MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgardLarge5MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgardLarge_5"
        super(ConstantHolmgardLarge5MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgardLarge6MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgardLarge_6"
        super(ConstantHolmgardLarge6MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgardLarge7MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgardLarge_7"
        super(ConstantHolmgardLarge7MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgardLarge8MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgardLarge_8"
        super(ConstantHolmgardLarge8MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgardLarge9MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgardLarge_9"
        super(ConstantHolmgardLarge9MdEnv, self).__init__(stage_name=stage_name)


class ConstantHolmgardLarge10MdEnv(MdEnvBase):
    """Constant Holmgard Class."""

    def __init__(self):
        stage_name: Final[str] = "ConstantHolmgardLarge_10"
        super(ConstantHolmgardLarge10MdEnv, self).__init__(stage_name=stage_name)

class Policy0MdEnv(MdEnvBase):
    """PolicyMdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "policy_0"
        super(Policy0MdEnv, self).__init__(stage_name=stage_name)

class Policy1MdEnv(MdEnvBase):
    """PolicyMdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "policy_1"
        super(Policy1MdEnv, self).__init__(stage_name=stage_name)

class Simple0MdEnv(MdEnvBase):
    """SimpleMdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "simple_0"
        super(Simple0MdEnv, self).__init__(stage_name=stage_name)

class Policy2MdEnv(MdEnvBase):
    """SwitchMdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "policy_2"
        super(Policy2MdEnv, self).__init__(stage_name=stage_name)

class Policy3MdEnv(MdEnvBase):
    """SwitchMdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "policy_3"
        super(Policy3MdEnv, self).__init__(stage_name=stage_name)

class Policy4MdEnv(MdEnvBase):
    """SwitchMdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "policy_4"
        super(Policy4MdEnv, self).__init__(stage_name=stage_name)

class Policy5MdEnv(MdEnvBase):
    """SwitchMdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "policy_5"
        super(Policy5MdEnv, self).__init__(stage_name=stage_name)

class Policy6MdEnv(MdEnvBase):
    """SwitchMdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "policy_6"
        super(Policy6MdEnv, self).__init__(stage_name=stage_name)

class Policy7MdEnv(MdEnvBase):
    """SwitchMdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "policy_7"
        super(Policy7MdEnv, self).__init__(stage_name=stage_name)

class Policy8MdEnv(MdEnvBase):
    """SwitchMdEnv Class."""

    def __init__(self):
        stage_name: Final[str] = "policy_8"
        super(Policy8MdEnv, self).__init__(stage_name=stage_name)

class PolicySwitch1MdEnv(MdPolicyEnv):
    """PolicyMdEnv Class."""

    def __init__(self, config:dict = default_config):
        stage_name: Final[str] = "policy_1"
        super(PolicySwitch1MdEnv, self).__init__(stage_name=stage_name, config= config)

class HolmgardSwitch9MdEnv(MdPolicyEnv):
    """Holmgard Class."""

    def __init__(self, config:dict = default_config):
        stage_name: Final[str] = "holmgard_9"
        super(HolmgardSwitch9MdEnv, self).__init__(stage_name=stage_name, config = config)

class PolicySwitch2MdEnv(MdPolicyEnv):
    """PolicyMdEnv Class."""

    def __init__(self, config:dict = default_config):
        stage_name: Final[str] = "policy_2"
        super(PolicySwitch2MdEnv, self).__init__(stage_name=stage_name, config = config)

class PolicySwitch3MdEnv(MdPolicyEnv):
    """Switch Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "policy_3"
        super(PolicySwitch3MdEnv, self).__init__(stage_name=stage_name,config = config)

class PolicySwitch4MdEnv(MdPolicyEnv):
    """Switch Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "policy_4"
        super(PolicySwitch4MdEnv, self).__init__(stage_name=stage_name,config = config)

class PolicySwitch5MdEnv(MdPolicyEnv):
    """Switch Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "policy_5"
        super(PolicySwitch5MdEnv, self).__init__(stage_name=stage_name,config = config)

class PolicySwitch6MdEnv(MdPolicyEnv):
    """Switch Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "policy_6"
        super(PolicySwitch6MdEnv, self).__init__(stage_name=stage_name,config = config)

class PolicySwitch7MdEnv(MdPolicyEnv):
    """Switch Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "policy_7"
        super(PolicySwitch7MdEnv, self).__init__(stage_name=stage_name,config = config)

class PolicySwitch8MdEnv(MdPolicyEnv):
    """Switch Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "policy_8"
        super(PolicySwitch8MdEnv, self).__init__(stage_name=stage_name,config = config)

class HolmgardSwitch0MdEnv(MdPolicyEnv):
    """Holmgard Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "holmgard_0"
        super(HolmgardSwitch0MdEnv, self).__init__(stage_name=stage_name,config = config)

class HolmgardSwitch1MdEnv(MdPolicyEnv):
    """Holmgard Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "holmgard_1"
        super(HolmgardSwitch1MdEnv, self).__init__(stage_name=stage_name,config = config)

class HolmgardSwitch2MdEnv(MdPolicyEnv):
    """Holmgard Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "holmgard_2"
        super(HolmgardSwitch2MdEnv, self).__init__(stage_name=stage_name,config = config)

class HolmgardSwitch3MdEnv(MdPolicyEnv):
    """Holmgard Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "holmgard_3"
        super(HolmgardSwitch3MdEnv, self).__init__(stage_name=stage_name,config = config)

class HolmgardSwitch4MdEnv(MdPolicyEnv):
    """Holmgard Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "holmgard_4"
        super(HolmgardSwitch4MdEnv, self).__init__(stage_name=stage_name,config = config)

class HolmgardSwitch5MdEnv(MdPolicyEnv):
    """Holmgard Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "holmgard_5"
        super(HolmgardSwitch5MdEnv, self).__init__(stage_name=stage_name,config = config)

class HolmgardSwitch5MdEnv(MdPolicyEnv):
    """Holmgard Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "holmgard_5"
        super(HolmgardSwitch5MdEnv, self).__init__(stage_name=stage_name,config = config)

class HolmgardSwitch6MdEnv(MdPolicyEnv):
    """Holmgard Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "holmgard_6"
        super(HolmgardSwitch6MdEnv, self).__init__(stage_name=stage_name,config = config)

class HolmgardSwitch7MdEnv(MdPolicyEnv):
    """Holmgard Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "holmgard_7"
        super(HolmgardSwitch7MdEnv, self).__init__(stage_name=stage_name,config = config)

class HolmgardSwitch8MdEnv(MdPolicyEnv):
    """Holmgard Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "holmgard_8"
        super(HolmgardSwitch8MdEnv, self).__init__(stage_name=stage_name,config = config)

class HolmgardSwitch9MdEnv(MdPolicyEnv):
    """Holmgard Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "holmgard_9"
        super(HolmgardSwitch9MdEnv, self).__init__(stage_name=stage_name,config = config)

class HolmgardSwitch10MdEnv(MdPolicyEnv):
    """Holmgard Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "holmgard_10"
        super(HolmgardSwitch10MdEnv, self).__init__(stage_name=stage_name,config = config)

class HardSwitchMdEnv(MdPolicyEnv):
    """HardMdEnv Class."""

    def __init__(self, config:dict= default_config):
        stage_name: Final[str] = "hard"
        super(HardSwitchMdEnv, self).__init__(stage_name=stage_name, config = config)

class CheckSwitch1MdEnv(MdPolicyEnv):
    """Check1MdEnv Class."""

    def __init__(self, config:dict= default_config):
        stage_name: Final[str] = "check_1"
        super(CheckSwitch1MdEnv, self).__init__(stage_name=stage_name, config = config)

class CheckSwitch2MdEnv(MdPolicyEnv):
    """Check1MdEnv Class."""

    def __init__(self, config:dict= default_config):
        stage_name: Final[str] = "check_2"
        super(CheckSwitch2MdEnv, self).__init__(stage_name=stage_name, config = config)

class CheckSwitch3MdEnv(MdPolicyEnv):
    """Check1MdEnv Class."""

    def __init__(self, config:dict= default_config):
        stage_name: Final[str] = "check_3"
        super(CheckSwitch3MdEnv, self).__init__(stage_name=stage_name, config = config)

class StrandSwitch1MdEnv(MdPolicyEnv):
    """Strand1MdEnv Class."""

    def __init__(self, config:dict= default_config):
        stage_name: Final[str] = "strand_1"
        super(StrandSwitch1MdEnv, self).__init__(stage_name=stage_name, config = config)

class StrandSwitch2MdEnv(MdPolicyEnv):
    """Strand1MdEnv Class."""

    def __init__(self, config:dict= default_config):
        stage_name: Final[str] = "strand_2"
        super(StrandSwitch2MdEnv, self).__init__(stage_name=stage_name, config = config)

class StrandSwitch3MdEnv(MdPolicyEnv):
    """Strand1MdEnv Class."""

    def __init__(self, config:dict= default_config):
        stage_name: Final[str] = "strand_3"
        super(StrandSwitch3MdEnv, self).__init__(stage_name=stage_name, config = config)

class StrandSwitch4MdEnv(MdPolicyEnv):
    """Strand1MdEnv Class."""

    def __init__(self, config:dict= default_config):
        stage_name: Final[str] = "strand_4"
        super(StrandSwitch4MdEnv, self).__init__(stage_name=stage_name, config = config)

class StrandSwitch5MdEnv(MdPolicyEnv):
    """Strand1MdEnv Class."""

    def __init__(self, config:dict= default_config):
        stage_name: Final[str] = "strand_5"
        super(StrandSwitch5MdEnv, self).__init__(stage_name=stage_name, config = config)

class GeneSwitch1MdEnv(MdPolicyEnv):
    """Random1Env Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "gene_1"
        super(GeneSwitch1MdEnv, self).__init__(stage_name=stage_name,config=config)


class GeneSwitch2MdEnv(MdPolicyEnv):
    """Random1Env Class."""

    def __init__(self,config:dict= default_config):
        stage_name: Final[str] = "gene_2"
        super(GeneSwitch2MdEnv, self).__init__(stage_name=stage_name,config=config)