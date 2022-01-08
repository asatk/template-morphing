
PHI_MASS=500
ROOT_PATH="/home/asatk/Documents/code/cern/TM/src"
SEED=2020
NSIM=3
NITERS=6000
N_DISTS=120
N_SAMPLES_TRAIN=10
STD_GAUSSIAN=0.02
RADIUS=1
BATCH_SIZE_D=128
BATCH_SIZE_G=128
LR_GAN=5e-5
SIGMA=-1.0
KAPPA=-2.0
DIM_GAN=2

echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN Hard"
CUDA_VISIBLE_DEVICES=0 python main.py --phi_mass $PHI_MASS --root_path $ROOT_PATH --GAN CcGAN --nsim $NSIM --seed $SEED --n_dists $N_DISTS --n_samples_train $N_SAMPLES_TRAIN --sigma_gaussian $STD_GAUSSIAN --radius $RADIUS --niters_gan $NITERS --resume_niters_gan 0 --lr_gan $LR_GAN --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --kernel_sigma $SIGMA --threshold_type hard --kappa $KAPPA --eval --dim_gan $DIM_GAN # 2>&1 | tee output_hard.txt

echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN Soft"
CUDA_VISIBLE_DEVICES=0 python main.py --phi_mass $PHI_MASS --root_path $ROOT_PATH --GAN CcGAN --nsim $NSIM --seed $SEED --n_dists $N_DISTS --n_samples_train $N_SAMPLES_TRAIN --sigma_gaussian $STD_GAUSSIAN --radius $RADIUS --niters_gan $NITERS --resume_niters_gan 0 --lr_gan $LR_GAN --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA --eval --dim_gan $DIM_GAN # 2>&1 | tee output_soft.txt