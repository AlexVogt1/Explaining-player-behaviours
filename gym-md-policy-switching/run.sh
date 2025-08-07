echo "------------------------------------------------------------------------"
echo "Job started on" `date`
echo "------------------------------------------------------------------------"
echo Running on $HOSTNAME...
echo Running on $HOSTNAME... >&2

reward_test_levels=("policy_6" "policy_7" "policy_8")
# reward_test_levels=("gene_1" "strand_2" "check_1" "holmgard_1" "hard")
levels=("policy_2" "policy_3" "policy_5" "policy_6" "policy_7" "policy_8" "holmgard_0" "holmgard_1" "holmgard_2" "holmgard_3" "holmgard_4" "holmgard_5" "holmgard_6" "holmgard_7" "holmgard_8" "holmgard_9" "holmgard_10" "strand_1" "strand_2" "strand_3" "strand_4" "strand_5" "check_1" "check_2" "check_3" "gene_1" "gene_2")
# levels=("holmgard_2" "holmgard_3" "holmgard_4" "holmgard_5" "holmgard_6" "holmgard_7" "holmgard_8" "holmgard_9" "check_1" "strand_2" "check_2" "check_3" "strand_1")
# levels=("hard" "holmgard_0" "holmgard_1" "holmgard_2" "holmgard_3" "check_1" "strand_2" "check_2" "check_3" "strand_1")
# levels=("check_1" "strand_2" "check_2" "check_3" "strand_1")
basic_play_styles=("killer" "runner" "potion")
analysis_play_styles=("switch" "hard" "treasure" "killer" "runner" "potion")
play_styles=("runner_safe" "runner_risky" "killer_safe" "killer_risky" "treasure_safe" "treasure_risky" "clearer_safe" "clearer_risky")

# source ~/.bashrc
# cd ~/src/EDPCGARL/gym-pcgrl

# for i in "${analysis_play_styles[@]}" 
# do
#     echo "$i"
#     python train_ppo.py --env=switch-hard --play_style="$i" --reward_scheme="$i" --exp_type=switch_analysis_killer --action_type=policy --action_space_type=discrete --obs_type=distance --algo=PPO --base_path=./play_style_models/base/
# done
for i in "${levels[@]}" 
do
    echo "$i"
    python train_ppo.py --env=switch-"$i" --play_style=switch --reward_scheme=switch --exp_type=groupShap_64_01 --action_type=policy --action_space_type=discrete --obs_type=distance --algo=PPO --base_path=./play_style_models/base/
done
# python train_ppo.py --env=switch-hard --play_style=killer --reward_scheme=killer --exp_type=switch_analysis_killer_net_bigbatch --action_type=policy --action_space_type=discrete --obs_type=distance --algo=PPO --base_path=./play_style_models/base/


# for i in "${levels[@]}"
# do
#     for j in "${analysis_play_styles[@]}" 
#     do
#         echo "$i" "$j"
#         python train_ppo.py --env=switch-"$i" --play_style="$j" --reward_scheme="$j" --exp_type=switch_analysis_new --action_type=policy --action_space_type=discrete --obs_type=distance --algo=PPO --base_path=./play_style_models/base/
#     done
# done
# python train_ppo.py --env=switch-policy_4 --play_style=runner --reward_scheme=runner --exp_type=grid_base_12x12 --action_type=path --action_space_type=box --obs_type=grid --algo=PPO

# # python train_ppo.py --env=switch-holmgard_9 --play_style=hard --reward_scheme=exitLessBetter4 --exp_type=switch --action_type=policy --algo=DQN

# # python train_ppo.py --env=switch-policy_2 --play_style=switch --reward_scheme=switch --exp_type=config_test --action_type=path --action_space_type=box --obs_type=grid --algo=PPO

# # grid base policies
# # for i in "${basic_play_styles}"; do
# #     python train_ppo.py --env=switch-policy_4 --play_style="$i" --reward_scheme="$i" --exp_type=grid_base_12x12 --action_type=path --action_space_type=box --obs_type=grid --algo=PPO
# # done
# for i in "${analysis_play_styles}"; do
#     python train_ppo.py --env=switch-hard --play_style="$i" --reward_scheme="$i" --exp_type=switch_analysis_killer --action_type=policy --action_space_type=discrete --obs_type=distance --algo=PPO --base_path=./play_style_models/base/
# done
# python train_ppo.py --env=switch-check_1 --play_style=potion --reward_scheme=potion --exp_type=switch_analysis --action_type=policy --action_space_type=discrete --obs_type=distance --algo=PPO --base_path=./play_style_models/base/
# for i in "${levels[@]}"; do
#     for j in "${analysis_play_styles[@]}"; do   # The quotes are necessary here
#         python train_ppo.py --env=switch-"$i" --play_style="$j" --reward_scheme="$j" --exp_type=switch_analysis --action_type=policy --action_space_type=discrete --obs_type=distance --algo=PPO --base_path=./play_style_models/base/
# done
# done
# for i in "${reward_test_levels[@]}"; do   # The quotes are necessary here
#     for j in "${basic_play_styles[@]}"; do   # The quotes are necessary here
#         python train_ppo.py --env="$i" --play_style="$j" --reward_scheme=fifty_twoFiftytwo --exp_type=SteprewardShaping
# done
# done


#train killer
# for i in "${reward_test_levels[@]}"; do   # The quotes are necessary here
#     python train_ppo.py --env="$i" --play_style=killer --reward_scheme=fiftytwoFifty --exp_type=rewardShapingKillerLonger
# done

#finish hard exp
# fin_hard=("treasure_risky" "clearer_safe" "clearer_risky")
# for i in "${fin_hard[@]}"; do   # The quotes are necessary here
#     python train_ppo.py --env=hard --play_style="$i" --reward_scheme=not_original --exp_type=baseline
# done

#holmgard_9 original baseline
# python train_ppo.py --env="holmgard_9" --exp_type=baseline

#run with original reward
# for i in "${levels[@]}"; do   # The quotes are necessary here
#     python train_ppo.py --env="$i" --exp_type=baseline
# done

#train all other play_styles over all listed levels
# for i in "${levels[@]}"; do   # The quotes are necessary here
#     for j in "${play_styles[@]}"; do   # The quotes are necessary here
#         python train_ppo.py --env="$i" --play_style="$j" --reward_scheme=not_original --exp_type=baseline
# done
# done

# conda activate test1
# which python
# echo "Hello, we are doing a job now, seg smb, wide"
# pwd
# python train.py
echo "------------------------------------------------------------------------"
echo "Job ended on" `date`
echo "------------------------------------------------------------------------"