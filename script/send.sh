#!/bin/bash

# tee 1
rsync -avzh . -e 'ssh -p 13741 -i /home/z/.ssh/id_siemens' root@ssh4.vast.ai:~/tee/ --progress --exclude={__pycache__,checkpoints,logs,.git,plot,data,debug,*gradcam*,*.png,*.jpg,*.JPG,*.PNG}

# rsync -avzh saved/data/fer2013 -e 'ssh -p 13741 -i /home/z/.ssh/id_siemens' root@ssh4.vast.ai:~/tee/saved/data/ --progress

# rsync -avzh saved/data/z/*.npy -e 'ssh -p 13741 -i /home/z/.ssh/id_siemens' root@ssh4.vast.ai:~/tee/saved/data/z/ --progress

#  rsync -avzh 'saved/checkpoints/Z_resmasking_dropout1_rot30_2019Nov30_13.32' -e 'ssh -p 13741 -i /home/z/.ssh/id_siemens' root@ssh4.vast.ai:~/tee/saved/checkpoints/ --progress
