import os
import torch
from safetensors.torch import save_file

def export(
    model: torch.float32, 
    model_name: str, 
    model_dir: str, 
    default_export: bool=True, 
    torch_script: bool=True, 
    onnx: bool=True, 
    safetensors: bool=True, 
    input_size: int=1
):
    """
    Exports a trained model from PyTorch to a few formats:
    
    1. Default PyTorch (default_export)
    2. TorchScript (torch_script)
    3. ONNX (onnx)
    4. Safetensors (safetensors)
    
    For more info on exports 1 and 2, see this link: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    
    For export 3, see: https://pytorch.org/tutorials//beginner/onnx/export_simple_model_to_onnx_tutorial.html
    
    For export 4, see: https://huggingface.co/docs/safetensors/en/index
    
    Parameters:
        model (torch.float32): The PyTorch model
        model_name (str): Name of the model
        model_dir (str): Directory to save the model
        default_export (bool): True to export the model in default PyTorch-only format, False to not. Defaults to True
        torch_script (bool): True to export model in TorchScript format, False to not. Defaults to True
        onnx (bool): True to export model in ONNX format, False to not. Defaults to True
        safetensors (bool): True to export model's state_dict in Safetensors format. Fale to not. Defaults to True
        input_size (int): Size of input data. Defaults to 1
    
    Returns:      
        None
    """
    # Set model to evaluation mode before exporting
    model.eval()
    
        # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Export as default PyTorch model state dict
    if default_export:
        default_name = os.path.join(model_dir, f'{model_name}.pth')
        torch.save(model.state_dict(), default_name)
        print(f'Model state_dict saved in PyTorch format to {default_name}')
    else:
        pass
    
    # Export model in TorchScript format
    if torch_script:
        model_scripted = torch.jit.script(model)
        scripted_name = os.path.join(model_dir, f'{model_name}_tscript.pt')
        model_scripted.save(scripted_name)
        print(f'Model saved in TorchScript format to {scripted_name}')
    else:
        pass
    
    # Export model in ONNX format
    if onnx:
        onnx_name = os.path.join(model_dir, f'{model_name}.onnx')
        dummy_input = torch.randn(1, input_size)
        torch.onnx.export(model, dummy_input, onnx_name)
        print(f'Model saved in ONNX format to {onnx_name}')
    else:
        pass
    
    # Export model in Safetensors format
    if safetensors:
        safetensors_name = os.path.join(model_dir, f'{model_name}.safetensors')
        state_dict = model.state_dict()
        save_file(model.state_dict(), safetensors_name)
        print(f'Model state_dict saved in Safetensors format to {safetensors_name}')
    else:
        pass