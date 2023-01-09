# Environment setup
Using files environment.yml to generate a new env. In Linux system, the command is:
`conda env create -f environment.yml`

Then you need to manually install adver-torch0.2.4. Downloading the advertorch-master dir in Featurize and run within advertorch-master:
`python setup.py install`

# Reminder
## 1. If cannot downloading the checkpoints for ldm model.
Try removing following code in attack_ldm_fix_path.py.
`import ssl
ssl._create_default_https_context = ssl._create_unverified_context`

## 2. If the framework is stopped because of some strange erorrs.
Try using follow commands to restart the framework, where the digital following -r is 2+current maximized output dir. For example, if the maxmized output dir in path "outputs/txt2img-samples-textual-inversion/danbooru2017/8_40_1.0/" is 116. Then r is chosen as 18.
The option '-no_del' and '-no_attack' is crucial to avoid mis-deleting existing files.
`python file_testing_framework.py  -out danbooru2017 -input ../datasets/danbooru2017 -t style -bs 4 --no_eval -no_del -no_attack -r 100`

# Commands
## Sheep dataset

Preparations for running the framework testing (You should create the dir ./datasets/sheep brfore running the commands):
`python clean_package.py  -out sheep -input ../datasets/lsun_dataset/sheep`


Running the testing (You should create the dir "outputs/txt2img-samples-textual-inversion" and dir "framework_logs" rfore running the commands):
`python file_testing_framework.py  -out sheep -input ../datasets/lsun_dataset/sheep -t object -bs 4 --no_eval `


Example textual_inversion_img2img (w/o attack):
`python scripts/img2img.py --init-img  source/object/summary_img/source_1.png --ddim_eta 0.0 --n_samples 2 --n_iter 2 --scale 10.0 --ddim_steps 50  --embedding_path logs/summary_img2022-12-03T19-18-20_Attack_Object1_8_40_1/checkpoints/embeddings_gs-4999.pt --ckpt models/ldm/text2img-large/model.ckpt --prompt "The * on the beach"`


Example textual_inversion_img2img (w attack):
`python scripts/img2img.py --init-img  source/object/summary_img/source_1.png --ddim_eta 0.0 --n_samples 2 --n_iter 2 --scale 10.0 --ddim_steps 50  --embedding_path logs/summary_img2022-12-03T18-59-14_Source_Object1/checkpoints/embeddings_gs-4999.pt --ckpt models/ldm/text2img-large/model.ckpt --prompt "The * on the beach"`

## Wikiart
Preparations for running the framework testing(You should create the dir ./datasets/danbooru2017 brfore running the commands):
`python clean_package.py  -out danbooru2017 -input ../datasets/danbooru2017`


Running the testing (You should create the dir "outputs/txt2img-samples-textual-inversion" and dir "framework_logs" rfore running the commands):
`python file_testing_framework.py  -out danbooru2017 -input ../datasets/danbooru2017 -t style -bs 4 --no_eval`
