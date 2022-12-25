Preparations for running the framework testing:
`python clean_package.py  -out sheep -input ../datasets/lsun_dataset/sheep`


Running the testing :
`python file_testing_framework.py  -out sheep -input ../datasets/lsun_dataset/sheep -t object -bs 4 --no_eval --no_attack`