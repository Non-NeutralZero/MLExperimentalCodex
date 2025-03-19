## Set-up
- I used the jupyter/datascience-notebook image for this project
- You can run it using the following command (Don't forget to specify the path in which the notebook is located)

```shell
sudo docker run -it --rm -p 10000:8888 -v <PATH>:/home/jovyan/work -e NB_UID=$(id -u) -e NB_GID=$(id -g) -e GRANT_SUDO=yes jupyter/datascience-notebook
```

- Use the link provided with the token to open JupyterLab, and change the port to 10000 (for example:  http://127.0.0.1:10000/lab?token=c22df6af1f07e191a136afbd03be80052ae181a9b874046b)
- Download the data from (the kaggle compitition page)[https://www.kaggle.com/competitions/home-credit-default-risk/data]
- Add it to /home/jovyan/work/home-credit-default-risk/

## Docker RUN

Run the docker container using the image from jupyter/datascience-notebook

``` shell
sudo docker run -it --rm -p 10000:8888 -v $HOME/<PATH>:/home/jovyan/work -e NB_UID=$(id -u) -e NB_GID=$(id -g) -e GRANT_SUDO=yes jupyter/datascience-notebook

```