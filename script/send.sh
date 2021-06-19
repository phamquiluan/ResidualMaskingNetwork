#!/bin/bash
rsync -avzhP . -e 'ssh -p 13741 -i /home/z/.ssh/id_rsa' root@ssh4.vast.ai:~/tee/ --exclude={__pycache__,checkpoints,logs,.git,plot,data,debug,*gradcam*,*.png,*.jpg,*.JPG,*.PNG}

rsync -avzh -e 'ssh -p 13741 -i /home/z/.ssh/id_rsa' root@ssh4.vast.ai:~/tee/saved/* saved/ --progress --exclude={.git,__pycache__,images}
