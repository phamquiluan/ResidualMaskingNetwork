#!/bin/bash

# rsync -avzh -e 'ssh -p 13741 -i /home/z/.ssh/id_siemens' root@ssh4.vast.ai:~/tee/gen_results.py . --progress --exclude={.git,__pycache__,images}

rsync -avzh -e 'ssh -p 13741 -i /home/z/.ssh/id_siemens' root@ssh4.vast.ai:~/tee/saved/* saved/ --progress --exclude={.git,__pycache__,images}

# rsync -avzh -e 'ssh -p 13741 -i /home/z/.ssh/id_siemens' root@ssh4.vast.ai:~/tee/wrong_in_vemo.zip . --progress --exclude={.git,__pycache__,images,*.png,*.jpg}
