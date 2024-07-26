import yaml
import sys

def generate_run_name(config):
    model_params = config['ModelArguments']['model_name_or_path'].split('/')[-1]
    if '8B' in model_params:
        model_size = '8b'
    elif '70B' in model_params:
        model_size = '70b'
    else:
        model_size = 'xb'
    
    model_name = 'llama3_' + model_size
    
    num_train_epochs = config['TrainingArguments']['num_train_epochs']
    visual_features = config['SignDataArguments']['visual_features']
    enabled_features = [key for key, val in visual_features.items() if val['enable_input']]
    visual_features_str = "_".join(enabled_features)
    
    use_paraphrases = '_par' if config['SignDataArguments']['use_paraphrases'] else ''
    label_smoothing = config['TrainingArguments'].get('label_smoothing_factor', 0)
    label_smoothing_str = f"_lsmoothing{int(label_smoothing*10):02}" if label_smoothing > 0 else ''
    
    run_name = f"{model_name}_pre_{num_train_epochs}ep_{visual_features_str}{use_paraphrases}{label_smoothing_str}"
    run_name = run_name.strip('_')
    
    return run_name

def update_output_dir(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    run_name = generate_run_name(config)
    config['TrainingArguments']['output_dir'] = f'signllava/checkpoints/{run_name}'
    
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: config2config.py <config_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    update_output_dir(file_path)
    print(f"Updated output_dir in {file_path} with run name.")
