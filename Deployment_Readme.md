## Deployment
0. Download the model from huggingface and save it in a root folder called saved_model- This step has been done for you
model_path: Koleshjr/dyu-fr-joeynmt-316-epochs_2_layers_5heads_128_300_plateau_2000_7_45_20_93

1. build the container locally and give it a tag
    docker build -t local/highwind-examples/dyu-fr-inference:latest .

## Local testing
1. After building the Kserve predictor image that contains your model, spin it up to test your model inference

    docker compose up -d
    docker compose logs

2. Finally, send a payload to your model to test its response. To do this, use the curl cmmand to send a POST request with an example JSON payload.

Run this from another terminal (remember to navigate to this folder first)

* Linux/Mac Bash/zsh

    curl -X POST http://localhost:8080/v2/models/model/infer -H 'Content-Type: application/json' -d @./input.json

* Windows PowerShell

    $json = Get-Content -Raw -Path ./input.json
    $response = Invoke-WebRequest -Uri http://localhost:8080/v2/models/model/infer -Method Post -ContentType 'application/json' -Body ([System.Text.Encoding]::UTF8.GetBytes($json))
    $responseObject = $response.Content | ConvertFrom-Json
    $responseObject | ConvertTo-Json -Depth 10

## Finally Push to High-Wind
1. Create an asset on High-Wind by following the steps prompted
2. Get the push commands from High-Wind then use them to push the image to high-wind