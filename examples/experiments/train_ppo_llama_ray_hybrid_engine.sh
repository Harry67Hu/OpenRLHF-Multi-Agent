set -x

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
    "working_dir": "/data/hutianyi-trl/OpenRLHF-Multi-Agent",
    "conda": {"name": "rlhf"}
}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 8 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 4 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.4 \
   --pretrain /data/hf_hub/models/OpenRLHF-Llama-3-8b-sft-mixture \
   --reward_pretrain /data/hf_hub/models/OpenRLHF-Llama-3-8b-rm-mixture \
   --save_path /data/hutianyi-trl/OpenRLHF-Multi-Agent/examples/test_scripts/final/llama3-8b-rlhf \
   --ckpt_path /data/hutianyi-trl/OpenRLHF-Multi-Agent/examples/test_scripts/ckpt/llama3-8b-rlhf \
   --save_hf_ckpt \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /data/hf_hub/datasets/OpenRLHF-prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep
