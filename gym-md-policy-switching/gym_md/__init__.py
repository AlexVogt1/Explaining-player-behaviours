"""gym-md init module."""
__version__ = "0.5.2"

from logging import NullHandler, getLogger

from gym.envs.registration import register

getLogger(__name__).addHandler(NullHandler())

register(
    id="md-simple_0-v0",
    entry_point="gym_md.envs:Simple0MdEnv",
)
register(
    id="md-policy_0-v0",
    entry_point="gym_md.envs:Policy0MdEnv",
)
register(
    id="md-policy_1-v0",
    entry_point="gym_md.envs:Policy1MdEnv",
)
register(
    id="md-switch-policy_1-v0",
    entry_point="gym_md.envs:PolicySwitch1MdEnv",
)
register(
    id="md-switch-policy_2-v0",
    entry_point="gym_md.envs:PolicySwitch2MdEnv",
)
register(
    id="md-switch-policy_3-v0",
    entry_point="gym_md.envs:PolicySwitch3MdEnv",
)
register(
    id="md-switch-policy_4-v0",
    entry_point="gym_md.envs:PolicySwitch4MdEnv",
)
register(
    id="md-switch-policy_5-v0",
    entry_point="gym_md.envs:PolicySwitch5MdEnv",
)
register(
    id="md-switch-policy_6-v0",
    entry_point="gym_md.envs:PolicySwitch6MdEnv",
)
register(
    id="md-switch-policy_7-v0",
    entry_point="gym_md.envs:PolicySwitch7MdEnv",
)
register(
    id="md-switch-policy_8-v0",
    entry_point="gym_md.envs:PolicySwitch8MdEnv",
)
register(
    id="md-policy_2-v0",
    entry_point="gym_md.envs:Policy2MdEnv",
)
register(
    id="md-policy_3-v0",
    entry_point="gym_md.envs:Policy3MdEnv",
)
register(
    id="md-policy_4-v0",
    entry_point="gym_md.envs:Policy4MdEnv",
)
register(
    id="md-policy_5-v0",
    entry_point="gym_md.envs:Policy5MdEnv",
)
register(
    id="md-policy_6-v0",
    entry_point="gym_md.envs:Policy6MdEnv",
)
register(
    id="md-policy_7-v0",
    entry_point="gym_md.envs:Policy7MdEnv",
)
register(
    id="md-policy_8-v0",
    entry_point="gym_md.envs:Policy8MdEnv",
)
register(
    id="md-base-v0",
    entry_point="gym_md.envs:MdEnvBase",
)
register(
    id="md-test-v0",
    entry_point="gym_md.envs:TestMdEnv",
)
register(
    id="md-edge-v0",
    entry_point="gym_md.envs:EdgeMdEnv",
)
register(
    id="md-hard-v0",
    entry_point="gym_md.envs:HardMdEnv",
)
register(
    id="md-random_1-v0",
    entry_point="gym_md.envs:Random1MdEnv",
)
register(
    id="md-random_2-v0",
    entry_point="gym_md.envs:Random2MdEnv",
)
for i in range(2):
    register(
        id=f"md-gene_{i + 1}-v0",
        entry_point=f"gym_md.envs:Gene{i + 1}MdEnv",
    )
for i in range(2):
    register(
        id=f"md-switch-gene_{i + 1}-v0",
        entry_point=f"gym_md.envs:GeneSwitch{i + 1}MdEnv",
    )
for i in range(5):
    register(
        id=f"md-strand_{i + 1}-v0",
        entry_point=f"gym_md.envs:Strand{i + 1}MdEnv",
    )
for i in range(3):
    register(
        id=f"md-check_{i + 1}-v0",
        entry_point=f"gym_md.envs:Check{i + 1}MdEnv",
    )
for i in range(11):
    register(
        id=f"md-holmgard_{i}-v0",
        entry_point=f"gym_md.envs:Holmgard{i}MdEnv",
    )
for i in range(11):
    register(
        id=f"md-constant-holmgard_{i}-v0",
        entry_point=f"gym_md.envs:ConstantHolmgard{i}MdEnv",
    )
for i in range(11):
    register(
        id=f"md-constant-holmgard-large_{i}-v0",
        entry_point=f"gym_md.envs:ConstantHolmgardLarge{i}MdEnv",
    )
for i in range(11):
    register(
        id=f"md-switch-holmgard_{i}-v0",
        entry_point=f"gym_md.envs:HolmgardSwitch{i}MdEnv",
    )

for i in range(3):
    register(
        id=f"md-switch-check_{i+1}-v0",
        entry_point=f"gym_md.envs:CheckSwitch{i+1}MdEnv",
    )

for i in range(5):
    register(
        id=f"md-switch-strand_{i + 1}-v0",
        entry_point=f"gym_md.envs:StrandSwitch{i + 1}MdEnv",
    )

register(
    id="md-switch-hard-v0",
    entry_point="gym_md.envs:HardSwitchMdEnv",
)
