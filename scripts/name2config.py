import yaml
import sys


def update_output_dir(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    run_name = str(file).split('/')[-1].split('.')[0]
    config['TrainingArguments']['output_dir'] = f'signllava/checkpoints/{run_name}'

    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: name2config.py <config_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    update_output_dir(file_path)
    print(f"Updated output_dir in {file_path} with run name.")
