import argparse

def calculate_dram_traffic(seq_len=2048, batch_size=4, lora_rank=64):
    num_layers = 32
    hidden_size = 4096
    
    # Trainable parameters per projection (LoRA A + LoRA B)
    params_per_proj = 2 * (hidden_size * lora_rank)
    total_trainable_params = num_layers * 4 * params_per_proj
    
    # Optimizer memory traffic per parameter:
    # Standard:
    # 1. Write gradient: 4 bytes (FP32)
    # 2. Read gradient: 4 bytes (FP32)
    # 3. Read active weight: 2 bytes (FP16)
    # 4. Read master weight: 4 bytes (FP32)
    # 5. Read moment m: 4 bytes (FP32)
    # 6. Read moment v: 4 bytes (FP32)
    # 7. Write master weight: 4 bytes (FP32)
    # 8. Write moment m: 4 bytes (FP32)
    # 9. Write moment v: 4 bytes (FP32)
    # 10. Write active weight: 2 bytes (FP16)
    traffic_per_param_standard = 4 + 4 + 2 + 4 + 4 + 4 + 4 + 4 + 4 + 2 # 32 bytes
    
    # Fused Backward AdamW:
    # Eliminates writing gradient to DRAM and reading it back in optimizer: saves 8 bytes/param
    traffic_per_param_fused = traffic_per_param_standard - 8 # 24 bytes
    
    # Activation traffic for backward pass:
    act_bytes_per_layer = batch_size * seq_len * hidden_size * 2
    total_act_traffic = num_layers * 4 * act_bytes_per_layer
    
    total_traffic_standard_bytes = (total_trainable_params * traffic_per_param_standard) + total_act_traffic
    total_traffic_fused_bytes = (total_trainable_params * traffic_per_param_fused) + total_act_traffic
    
    standard_gb = total_traffic_standard_bytes / (1024 ** 3)
    fused_gb = total_traffic_fused_bytes / (1024 ** 3)
    
    optimizer_reduction = (traffic_per_param_standard - traffic_per_param_fused) / traffic_per_param_standard * 100.0
    total_reduction = (standard_gb - fused_gb) / standard_gb * 100.0
    
    print(f"Total trainable LoRA parameters: {total_trainable_params:,}")
    print(f"Optimizer traffic per param: Standard={traffic_per_param_standard} B, Fused={traffic_per_param_fused} B")
    print(f"Optimizer-only DRAM traffic reduction: {optimizer_reduction:.1f}%")
    print(f"Standard approach estimated total DRAM traffic: {standard_gb:.3f} GB/step")
    print(f"Fused approach estimated total DRAM traffic:    {fused_gb:.3f} GB/step")
    print(f"Net DRAM traffic reduction:                    {total_reduction:.2f}%")

if __name__ == "__main__":
    calculate_dram_traffic()

