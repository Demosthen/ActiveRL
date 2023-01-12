FROM alejandrocn7/sinergym:latest

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh \
    && chmod +x ./Miniconda3-py310_22.11.1-1-Linux-x86_64.sh \
    && bash ./Miniconda3-py310_22.11.1-1-Linux-x86_64.sh -b -p /home/miniconda \
    && rm ./Miniconda3-py310_22.11.1-1-Linux-x86_64.sh

WORKDIR /home

RUN git clone https://github.com/Demosthen/ActiveRL.git && ls

WORKDIR /home/ActiveRL

RUN ["/bin/bash", "-c", "ls && source /home/miniconda/bin/activate \
    && conda init \
    && conda env create -f environment.yml \
    && conda activate ActiveRL \ 
    && pip install git+https://github.com/cooper-org/cooper.git\
    && pip install -e gym-simplegrid/ --no-deps\
    && pip install moviepy==1.0.3\
    && pip uninstall pygame -y \
    && pip install sinergym[extras]\
    && pip install gym==0.24.1\
    && pip install -e gym-simplegrid\
    && pip install dm_control==1.0.9"]