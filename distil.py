import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from datasets import load_dataset
from typing import Dict, List
import gc
import flora_opt
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
        
        formatted_text = "".join([f"Human: {turn['value']}\n\nAssistant: {next((t['value'] for t in conversation[i+1:] if t['from'] == 'gpt'), '')}\n\n" 
                                  for i, turn in enumerate(conversation) if turn['from'] == 'human'])

        encoded = self.tokenizer(formatted_text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }

def load_openhermes_data(tokenizer: AutoTokenizer, max_length: int, batch_size: int):
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    openhermes_dataset = OpenHermesDataset(dataset, tokenizer, max_length)
    dataloader = DataLoader(openhermes_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
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

        outputs = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = top_k_indices == i
            if mask.any():
                expert_input = x[mask]
                expert_output = expert(expert_input)
                outputs[mask] += expert_output * top_k_probs[mask][:, i].unsqueeze(-1)

        return outputs

class MoELlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, num_experts: int, num_active_experts: int):
        super().__init__(config)
        
        for layer in self.model.layers:
            input_dim = layer.mlp.gate_proj.in_features
            output_dim = layer.mlp.up_proj.out_features
            layer.mlp = MoELayer(input_dim, output_dim, num_experts, num_active_experts)

def create_moe_model(model_name: str, num_experts: int, num_active_experts: int, device: torch.device) -> MoELlamaForCausalLM:
    print("Loading configuration...")
    config = LlamaConfig.from_pretrained(model_name)
    
    print("Creating MoE model...")
    moe_model = MoELlamaForCausalLM(config, num_experts, num_active_experts)
    
    print("Loading original model...")
    original_model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    print("Transferring weights...")
    with torch.no_grad():
        for name, param in moe_model.named_parameters():
            if "mlp" not in name:
                param.data.copy_(original_model.state_dict()[name].to(dtype=torch.float32))
        
        for name, module in moe_model.named_modules():
            if isinstance(module, MoELayer):
                original_mlp_prefix = name.replace("mlp", "mlp.gate_proj")
                original_weight = original_model.state_dict()[f"{original_mlp_prefix}.weight"].to(dtype=torch.float32)
                
                for expert in module.experts:
                    expert.weight.data.copy_(original_weight)
                    if hasattr(expert, 'bias') and expert.bias is not None:
                        expert.bias.data.zero_()  # Initialize bias to zero if it exists
                
                module.router.weight.data.normal_(mean=0.0, std=0.02)
                if hasattr(module.router, 'bias') and module.router.bias is not None:
                    module.router.bias.data.zero_()
    
    print("Cleaning up...")
    del original_model
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Moving MoE model to device...")
    # Move model to GPU in chunks
    for name, param in moe_model.named_parameters():
        param.data = param.data.to(device)
        torch.cuda.empty_cache()
    
    return moe_model

@torch.no_grad()
def get_original_hidden_states(original_model, input_ids, attention_mask):
    original_outputs = original_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    return [h.detach() for h in original_outputs.hidden_states]

def train_moe_model(original_model: LlamaForCausalLM, moe_model: MoELlamaForCausalLM, 
                    train_dataloader: DataLoader, num_epochs: int, 
                    device: torch.device) -> None:
    original_model.to(device)
    moe_model.to(device)
    
    for param in original_model.parameters():
        param.requires_grad = False
    
    for layer in moe_model.model.layers:
        for param in layer.self_attn.parameters():
            param.requires_grad = False
    
    optimizer = flora_opt.Flora(moe_model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        moe_model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            original_hidden_states = get_original_hidden_states(original_model, input_ids, attention_mask)
            moe_outputs = moe_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            loss = sum(torch.nn.functional.mse_loss(orig_hidden, moe_hidden) 
                       for orig_hidden, moe_hidden in zip(original_hidden_states, moe_outputs.hidden_states))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {total_loss / len(train_dataloader)}")

def main():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    num_experts = 8
    num_active_experts = 2
    max_length = 512
    batch_size = 2
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Creating MoE model...")
    moe_model = create_moe_model(model_name, num_experts, num_active_experts, device)
    print("MoE model created!")

    print("Loading original model for comparison...")
    original_model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    print("Original model loaded!")

    print("Preparing dataset...")
    train_dataloader = load_openhermes_data(tokenizer, max_length, batch_size)
    print("Dataset prepared!")

    print("Starting training...")
    train_moe_model(original_model, moe_model, train_dataloader, num_epochs, device)
    print("Training completed!")

if __name__ == "__main__":
    main()