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
    image_size: int=512
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
        default_export (bool): True to export the model in default PyTorch-only format (e.g. model's state_dict), False to not. Defaults to True
        torch_script (bool): True to export entire model in TorchScript format, False to not. Defaults to True
        onnx (bool): True to export entire model in ONNX format, False to not. Defaults to True
        safetensors (bool): True to export model's state_dict in Safetensors format. Fale to not. Defaults to True
        image_size (int): Size of input image in pixels. Used to create dummy input for ONNX export, assuming square input size (image is image_size x image_size pixels). Defaults to 512
    
    Returns:      
        None
    """
    # Set model to evluation mode before exporting
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
        model_to_export = model.model.to(torch.float32).cpu()
        model_to_export.eval()
        
        dummy_input = torch.randn(1, 1, image_size, image_size, dtype=torch.float32, device='cpu')
        torch.onnx.export(
            model_to_export, 
            dummy_input,
            onnx_name,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['logk'],
            dynamic_axes={
                'input':  {0: 'batch_size'},     # allow batching
                'logk':   {0: 'batch_size'}
            }
                        )
        print(f'Model saved in ONNX format to {onnx_name}')
    else:
        pass
    
    # Export model in Safetensors format
    if safetensors:
        safetensors_name = os.path.join(model_dir, f'{model_name}.safetensors')
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        for k, v in state_dict.items():
            if v.dtype == torch.bfloat16:
                state_dict[k] = v.float()
                
        save_file(state_dict, 
                  safetensors_name)
        print(f'Model state_dict saved in Safetensors format to {safetensors_name}')
    else:
        pass
