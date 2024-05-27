# LLaVA on PHOEXIT14T
**1. Clone this repository, switch to `phoenix` branch, and navigate to `LLaVA` folder.**
```
git clone https://github.com/Este1le/Sign_LLaVA.git
git checkout phoenix
cd LLaVA
```
**2. Install the conda environment `llava`.**
```
conda env create --file environment.yml
conda activate llava
```
**3. Prepare data.**
Download data from SynologyDrive `phoenix14t/data.zip`, move the unzipped folder `data` to `llava/phoenix14t/data`.

**4. Run experiments.**
```
cd LLaVA
```
Pretraining:
```
sh phoenix14t/scripts/pretrain_xformers.sh 
```
Fine-tuning:
```
sh phoenix14t/scripts/finetune_lora.sh
```