# Cloud setup (Docker)
Instructions to setup environment and agent on a machine (e.g. cloud).

The agent image can be pulled with:

```$ docker pull kaikun213/my-neural-net ```

The environment image can be pulled with the command:

```$ docker pull kaikun213/my-world-of-bits ```

Afterwards simply start a docker container daemon with in shell mode:

```$ docker run --privileged --rm -it -d -e DOCKER_NET_HOST=172.17.0.1 -v /var/run/docker.sock:/var/run/docker.sock -v /home/kaikun/my_neural_net/src/:/usr/local/universe/my_neural_net/src --name "dynamic_NewSensoryLayer" kaikun213/my-neural-net:0.1 bash ```

Where ` -v /home/kaikun/my_neural_net/src/:/usr/local/universe/my_neural_net/src ` is mounting the server directory. This is not necessarily needed. The process is not yet optimized for running multiple experiments/containers in parallel.

And cd into the src folder:
```$ cd /usr/local/universe/my_neural_net/src ```

Here the python script can be run:
```$ python myExample.py ```

It will start the agent which launches a docker container with the environment.
To exit the docker daemon without stopping it press STR+P and STR+Q.
To re-attach just write `docker attach CONTAINER-ID`.
It can be stopped with `docker stop CONTAINER-ID` or writing `exit` in the container bash.

All docker images can be seen with:
``` docker images ```

And all currently running docker containers including the VNC ports to connect to the environment with:
``` docker ps -a ```

Connecting via VNC to observe an experiment simply with some software (e.g. Remmina Remote Desktop Client Ubuntu):
``` HOSTNAME:PORT ```
The password to connect is `openai`.


### Copy or modify
Files from a running container (e.g. results) can be copied with:

```docker cp CONTAINER-ID:/usr/local/universe/my_neural_net/src/results ./results ```

Example how new files can be added to the docker image (copy them, or if remove original first if replacement):

``` docker cp $MY_NEURAL_NET/src CONTAINER-ID:/usr/local/universe/my_neural_net/src ```

And after leave the running container with STR-P and STR-Q to commit the changes to the image:

```docker commit -m "COMMIT_MESSAGE" CONTAINER-ID my-neural-net:0.1 ```

Or to rebuild the image:
`docker build -t my-neural-net .`.

To publish it online:
Tag it `docker tag my-neural-net:0.1 $DOCKER_ID/my-neural-net:0.1` and push `docker push DOCKER_ID/my-neural-net:0.1`
