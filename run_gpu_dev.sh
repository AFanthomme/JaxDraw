# Does the build automatically
docker run --rm --gpus all --mount type=bind,src=./results,dst=/app/results -it $(docker build --build-arg SKIP_TESTS=true -f Dockerfile.gpu -q .) "$@" 