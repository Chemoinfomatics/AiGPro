# FROM python:3.11 as base
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS base
#non interactive mode
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && \
    apt-get install -y python3-pip python3-dev wget && \
    rm -rf /var/lib/apt/lists/*

COPY environment.yml environment.yml

RUN wget -O /tmp/anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh \
    && bash /tmp/anaconda.sh -b -p /anaconda \
    && eval "$(/anaconda/bin/conda shell.bash hook)" \
    && conda init \
    && conda update -n base -c defaults conda \
    && conda create --name env  \
    && conda activate env 


WORKDIR /home/aigpro

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt
RUN apt update && apt -y upgrade



# development image.
FROM base AS development
# RUN apt-get update && \
#     apt-get install -y python3-pip python3-dev && \
#     rm -rf /var/lib/apt/lists/*
# RUN wget -O /tmp/anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh \
#     && bash /tmp/anaconda.sh -b -p /anaconda \
#     && eval "$(/anaconda/bin/conda shell.bash hook)" \
#     && conda init \
#     && conda update -n base -c defaults conda \
#     && conda create --name env \
#     && conda activate env 

# COPY environment.yml environment.yml
# COPY requirements.txt requirements.txt
# RUN pip install -r requirements_dev.txt
# RUN pip install -r requirements.txt




# testing image.
FROM base AS testing
# RUN apt-get update && \
#     apt-get install -y python3-pip python3-dev && \
#     rm -rf /var/lib/apt/lists/*
# RUN wget -O /tmp/anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh \
#     && bash /tmp/anaconda.sh -b -p /anaconda \
#     && eval "$(/anaconda/bin/conda shell.bash hook)" \
#     && conda init \
#     && conda update -n base -c defaults conda \
#     && conda create --name env \
#     && conda activate env 

RUN pip install pytest && pip install requests



# production image.
FROM base AS production
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

COPY environment.yml environment.yml


# RUN wget -O /tmp/anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh \
    # && bash /tmp/anaconda.sh -b -p /anaconda \
RUN eval "$(/anaconda/bin/conda shell.bash hook)" \
    && conda init \
    && conda update -n base -c defaults conda \
    && conda create --name env  \
    && conda activate env 
# RUN conda activate env

WORKDIR /production

COPY . .
RUN pip install  .
RUN pip install Werkzeug

# ARG PORT=8097
# ARG HOST=0.0.0.0
# ARG APP_MODULE=aigpro.app:app
# ARG WORKERS_PER_CORE=1

ENV MODE=production
ENV APP_MODULE=${APP_MODULE}
ENV WORKERS_PER_CORE=${WORKERS_PER_CORE}
ENV HOST=${HOST}
ENV PORT=${PORT}
ENV TZ="Seoul/Asia"
EXPOSE ${PORT}


RUN echo ls -l
RUN chmod +x ./scripts/start.sh


ENTRYPOINT [ "./scripts/start.sh" ]
# RUN echo "gunicorn -w $(expr $(nproc) \* ${WORKERS_PER_CORE}) -k uvicorn.workers.UvicornWorker -b ${HOST}:${PORT} ${APP_MODULE}"
# ENTRYPOINT ["tail"]
# CMD ["-f","/dev/null"]