import os
import re

datasets = ['2', '3', '4', '6', '7', '9', '12', '13', '14', '15', '17', '18', '20']

app_path = r'software\\variational-depth-from-focus-master\\Release\\vdff.exe'
input_dir = 'datasets'
output_dir = 'results'

params = {
    'denomRegu': 0.1,
    'lambda': 0.5,
    'nrIterations': 50
}

for dataset_name in datasets:
    os.system(f'{app_path} \
        -dir {input_dir}\\{dataset_name} \
        -pageLocked 0 -denomRegu {params["denomRegu"]} -lambda {params["lambda"]} -nrIterations {params["nrIterations"]}\
        -export {output_dir}\\{dataset_name}\\Output-VDFF-{params["denomRegu"]}-{params["lambda"]}-{params["nrIterations"]}.png')
