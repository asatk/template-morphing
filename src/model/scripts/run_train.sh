
AXIS="phi"
PHI_MASS=500
# OMEGA_MASS=0.550
OMEGA_MASS=550
CONST_MASS=$OMEGA_MASS
ROOT_PATH="/home/asatk/Documents/code/cern/TM/src/model"
SEED=2020
NSIM=3
NITERS=5000
N_DISTS=120
N_SAMPLES_TRAIN=10000
N_SAMPLES_EVAL=10000
STD_GAUSSIAN=0.02
BATCH_SIZE_D=128
BATCH_SIZE_G=128
LR_GAN=5e-5
SIGMA_PHI=-1.0
# SIGMA_OMEGA=$(SIGMA_PHI/1000)
SIGMA_OMEGA=-1.0
KAPPA_PHI=-2.0
# KAPPA_OMEGA=$(KAPPA_PHI/1000)
KAPPA_OMEGA=-1.0
# SIGMA=0.25
# KAPPA=0.5
DIM_GAN=2

echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN Hard"
CUDA_VISIBLE_DEVICES=0 python model/main.py --axis $AXIS --const_mass $CONST_MASS --root_path $ROOT_PATH --GAN CcGAN --nsim $NSIM --seed $SEED --n_dists $N_DISTS --n_samples_train $N_SAMPLES_TRAIN --sigma_gaussian $STD_GAUSSIAN --niters_gan $NITERS --resume_niters_gan 0 --lr_gan $LR_GAN --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --kernel_sigma_phi $SIGMA_PHI --kernel_sigma_omega $SIGMA_OMEGA --threshold_type hard --kappa_phi $KAPPA_PHI --kappa_omega $KAPPA_OMEGA --eval --dim_gan $DIM_GAN # 2>&1 | tee output_hard.txt

# echo "-------------------------------------------------------------------------------------------------"
# echo "CcGAN Soft"
# CUDA_VISIBLE_DEVICES=0 python model/main.py --axis $AXIS --const_mass $OMEGA_MASS --root_path $ROOT_PATH --GAN CcGAN --nsim $NSIM --seed $SEED --n_dists $N_DISTS --n_samples_train $N_SAMPLES_TRAIN --sigma_gaussian $STD_GAUSSIAN --niters_gan $NITERS --resume_niters_gan 0 --lr_gan $LR_GAN --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA --eval --dim_gan $DIM_GAN # 2>&1 | tee output_soft.txt
