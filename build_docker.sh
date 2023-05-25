#!/bin/bash
sudo chmod 666 /var/run/docker.sock

project_name="linh0901/wifi_simulation_api"
container_name="test"

echo "docker build/push image: $project_name"
docker build -t $project_name .
# docker push $project_name

# echo "docker stop --filter "name=$container_name""
# docker stop $(sudo docker ps -f "name=$container_name" -q)
# docker rm $(sudo docker ps -a -f "name=$container_name" -q)

# echo "docker rmi --filter "dangling=true""
# docker rmi $(docker images --filter "dangling=true" -q)

# echo "docker run --name $container_name -p 8000:8000 $project_name"
# docker run --name $container_name -p 8000:8000 $project_name