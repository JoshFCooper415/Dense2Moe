import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from datasets import load_dataset
from typing import Dict, List
import copy
import gc

class OpenHermesDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversation = item['conversations']
        
        formatted_text = ""
        for turn in conversation:
            if turn['from'] == 'human':
                formatted_text += f"Human: {turn['value']}\n\n"
            elif turn['from'] == 'gpt':
                formatted_text += f"Assistant: {turn['value']}\n\n"

        encoded = self.tokenizer(formatted_text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }

def load_openhermes_data(tokenizer: AutoTokenizer, max_length: int, batch_size: int):
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
    openhermes_dataset = OpenHermesDataset(dataset, tokenizer, max_length)
    dataloader = DataLoader(openhermes_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return dataloader

class MoELayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_experts: int, num_active_experts: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts

        self.router = torch.nn.Linear(input_dim, num_experts)
        self.experts = torch.nn.ModuleList([torch.nn.Linear(input_dim, output_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_probs = torch.softmax(self.router(x), dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.num_active_experts, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        outputs = torch.einsum('bne,bec->bnc', top_k_probs, expert_outputs[top_k_indices])

        return outputs

class MoELlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, num_experts: int, num_active_experts: int):
        super().__init__(config)
        
        # Replace MLP layers with MoE layers
        for layer in self.model.layers:
            input_dim = layer.mlp.gate_proj.in_features
            output_dim = layer.mlp.up_proj.out_features
            layer.mlp = MoELayer(input_dim, output_dim, num_experts, num_active_experts)

def create_moe_model(model_or_name: str, num_experts: int, num_active_experts: int, device: torch.device) -> MoELlamaForCausalLM:
    if isinstance(model_or_name, str):
        # If a string is provided, assume it's a model name or path
        config = LlamaConfig.from_pretrained(model_or_name)
        original_model = LlamaForCausalLM.from_pretrained(model_or_name, device_map="auto", torch_dtype=torch.float16)

    # Create the MoE model with the loaded configuration
    moe_model = MoELlamaForCausalLM(config, num_experts, num_active_experts).to(device)
    
    # Transfer weights from the original model to the MoE model
    moe_model_dict = moe_model.state_dict()
    original_state_dict = original_model.state_dict()
    
    for name, param in original_state_dict.items():
        if name in moe_model_dict and "mlp" not in name:
            moe_model_dict[name].copy_(param)
    
    # Initialize MoE layers
    for name, module in moe_model.named_modules():
        if isinstance(module, MoELayer):
            original_mlp_prefix = name.replace("mlp", "mlp.gate_proj")
            for i, expert in enumerate(module.experts):
                expert.weight.data.copy_(original_state_dict[f"{original_mlp_prefix}.weight"])
                if hasattr(expert, 'bias') and expert.bias is not None:
                    expert.bias.data.copy_(original_state_dict[f"{original_mlp_prefix}.bias"])
            
            module.router.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module.router, 'bias') and module.router.bias is not None:
                module.router.bias.data.zero_()
    
    # Clear CUDA cache and garbage collect
    torch.cuda.empty_cache()
    gc.collect()
    
    return moe_model

def train_moe_model(original_model: LlamaForCausalLM, moe_model: MoELlamaForCausalLM, 
                    train_dataloader: DataLoader, num_epochs: int, 
                    device: torch.device) -> None:
    original_model.to(device)
    moe_model.to(device)
    
    # Freeze original model
    for param in original_model.parameters():
        param.requires_grad = False
    
    # Freeze attention layers in MoE model
    for layer in moe_model.model.layers:
        for param in layer.self_attn.parameters():
            param.requires_grad = False
    
    optimizer = torch.optim.AdamW(moe_model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        moe_model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass through both models
            with torch.no_grad():
                original_outputs = original_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            moe_outputs = moe_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Compute loss based on hidden states
            loss = 0
            for orig_hidden, moe_hidden in zip(original_outputs.hidden_states, moe_outputs.hidden_states):
                loss += torch.nn.functional.mse_loss(orig_hidden, moe_hidden)
            
            # Backpropagate and update MoE model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {total_loss / len(train_dataloader)}")

# Usage example
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
num_experts = 8
num_active_experts = 2
max_length = 512
batch_size = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
original_model = LlamaForCausalLM.from_pretrained(model_name)
print('Llama created!')
moe_model = create_moe_model(original_model, num_experts, num_active_experts,device)
print('MOE created!')
train_dataloader = load_openhermes_data(tokenizer, max_length, batch_size)
print('training start!')
train_moe_model(original_model, moe_model, train_dataloader, num_epochs=10, device=device)