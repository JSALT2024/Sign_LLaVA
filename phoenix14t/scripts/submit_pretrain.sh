JOBNAME=llama2
script=/exp/xzhang/slt/jsalt2024/LLaVA/phoenix14t/scripts/pretrain_xformers.sh
qsub -l mem_free=50G,h_rt=50:00:00,num_proc=2,gpu=3 -q gpu.q@@RTX -o /home/hltcoe/xzhang/log/llava/${JOBNAME}.log.o -e /home/hltcoe/xzhang/log/llava/${JOBNAME}.log.e -N ${JOBNAME} ${script}