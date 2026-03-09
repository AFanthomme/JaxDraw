docker run  --gpus all --rm --volume ./experiments:/app/experiments \
--rm --volume ./tests:/app/tests \
--mount type=bind,src=./results,dst=/app/results \
--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it $(docker build -q .) "$@" 