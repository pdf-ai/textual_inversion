Preparations for running the framework testing:
`python clean_package.py  -out sheep -input ../datasets/lsun_dataset/sheep`


Running the testing :
`python file_testing_framework.py  -out sheep -input ../datasets/lsun_dataset/sheep -t object -bs 4 --no_eval --no_attack`


Example textual_inversion_img2img (w/o attack):
`python scripts/img2img.py --init-img  source/object/summary_img/source_1.png --ddim_eta 0.0 --n_samples 2 --n_iter 2 --scale 10.0 --ddim_steps 50  --embedding_path logs/summary_img2022-12-03T19-18-20_Attack_Object1_8_40_1/checkpoints/embeddings_gs-4999.pt --ckpt models/ldm/text2img-large/model.ckpt --prompt "The * on the beach"`


Example textual_inversion_img2img (w attack):
`python scripts/img2img.py --init-img  source/object/summary_img/source_1.png --ddim_eta 0.0 --n_samples 2 --n_iter 2 --scale 10.0 --ddim_steps 50  --embedding_path logs/summary_img2022-12-03T18-59-14_Source_Object1/checkpoints/embeddings_gs-4999.pt --ckpt models/ldm/text2img-large/model.ckpt --prompt "The * on the beach"
`