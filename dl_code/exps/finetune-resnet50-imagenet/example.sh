/home/lin/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet50 --optimizer parallel_choco \
    --avg_model True --experiment test \
    --data imagenet --use_lmdb_data True --data_dir /mlodata1/tlin/dataset/ILSVRC/ --pin_memory True \
    --batch_size 512 --base_batch_size 256 --num_workers 2 \
    --num_epochs 90 --partition_data random --reshuffle_per_epoch True --stop_criteria epoch \
    --on_cuda True --n_mpi_process 8 --n_sub_process 4 --world 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3 \
    --lr 0.1 --lr_scaleup True --lr_scaleup_factor graph --lr_warmup True --lr_warmup_epochs 5 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 --lr_milestones 30,60,80 \
    --weight_decay 1e-4 --use_nesterov True --momentum_factor 0.9 \
    --comm_op sign --consensus_stepsize 0.5 \
    --hostfile hostfile --graph_topology ring --track_time True --track_detailed_time True  --display_tracked_time True \
    --python_path /home/lin/conda/envs/pytorch-py3.6/bin/python --mpi_path /home/lin/.openmpi/ --summary_freq 100 \
    --backend mpi --work_dir /mlodata1/tlin/decentralized_code --remote_exec False --clean_python False --mpi_env LD_LIBRARY_PATH=/home/lin/.openmpi/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
