VER=1.0
docker build --network=host -f docker/Dockerfile -t graspdatagen:$VER -t graspdatagen:latest .
