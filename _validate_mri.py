from pathlib import Path
import sys

sys.path.insert(0, r'c:\ALZHEIMER_DETECTION\app\api')
import main

main.registry.load()

samples = {
    'MildDemented': r'c:\ALZHEIMER_DETECTION\data\raw\MRI\Alzhiemer\combined_images\MildDemented\01259f4a-2b71-4571-b6de-026d7827292f.jpg',
    'ModerateDemented': r'c:\ALZHEIMER_DETECTION\data\raw\MRI\Alzhiemer\combined_images\ModerateDemented\01850009-6a02-4307-8092-0e3ab8f1f8b2.jpg',
    'NonDemented': r'c:\ALZHEIMER_DETECTION\data\raw\MRI\Alzhiemer\combined_images\NonDemented\00190c87-a639-417d-a6d9-bae6a9e68207.jpg',
    'VeryMildDemented': r'c:\ALZHEIMER_DETECTION\data\raw\MRI\Alzhiemer\combined_images\VeryMildDemented\0088f79a-9a91-4619-a16d-0092d992a723.jpg',
}

for label, path in samples.items():
    response = main._predict_mri(Path(path).read_bytes())
    print(f'{label} -> {response.predicted_class} | confidence={response.confidence} | probs={response.probabilities}')
