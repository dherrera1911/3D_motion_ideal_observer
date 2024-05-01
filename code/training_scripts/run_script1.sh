#!/bin/bash                                                                         
#SBATCH --job-name=motion_depth                                                     
#SBATCH --ntasks=1                                                                  
#SBATCH --mem=8G                                                                    
#SBATCH --time=00:40:00                                                             
#SBATCH --mail-type=ALL                                                             
#SBATCH --output=motion_analysis.out                                                
                                                                                    
# de acuerdo a lo que quiera ejecutar puede elegir entre las siguientes tres líneas.
#SBATCH --gres=gpu:1 # Can be gpu:p100:1 or gpu:a100:1                              
                                                                                    
#SBATCH --partition=normal                                                          
#SBATCH --qos=gpu                                                                   
                                                                                    
# Initialize conda for the Bash shell                                               
python ./1_train_3D_speed.py                                  

