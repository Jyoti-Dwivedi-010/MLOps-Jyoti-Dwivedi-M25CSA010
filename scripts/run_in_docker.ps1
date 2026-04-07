$null = docker info 2>$null
if ($LASTEXITCODE -ne 0) {
	Write-Host "Docker daemon is not running. Start Docker Desktop first, then rerun this script."
	exit 1
}

docker build -t assignment5-mlops .
docker run --gpus all --rm -it -v "$($PWD.Path):/workspace" assignment5-mlops bash
