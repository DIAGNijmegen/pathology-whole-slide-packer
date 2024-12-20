FROM ubuntu:22.04

ARG PYTHON_VERSION=3.10.12

# === Overwrite NVIDIA's authorship label
LABEL maintainer="Radboudumc"

STOPSIGNAL SIGINT

# === Configurate environment variables ===
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# === Install build packages ===
ENV DEBIAN_FRONTEND="noninteractive"

# === Install libs ===
# libgeos is necessary for shapely
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        ca-certificates apt-transport-https \
        sudo \
        git subversion \
        g++ ninja-build \
        vim \
		meson \
        wget curl \
        bzip2 zip unzip \
        dcmtk libboost-all-dev \
        libgomp1 \
        libjpeg-turbo8 \
        libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev gcovr libffi-dev uuid-dev \
        libgtk2.0-dev libgsf-1-dev libtiff5-dev libopenslide-dev \
        libgl1-mesa-glx libgeos-dev \
        libpugixml1v5 \
        build-essential pkg-config libglib2.0-dev libexpat1-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# line build-essential.. is for libvips, unzip curl is for awscli

# Set environment variables for CUDA 11.7
ENV PATH=/usr/local/cuda-11.7/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH


# === Configure timezone ===
RUN echo "Europe/Berlin" > /etc/timezone && \
    rm -f /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata


#  === Install latest Pixman ===
#  This should fix the bug described here (https://github.com/TissueImageAnalytics/tiatoolbox/issues/133)
RUN wget -q https://www.cairographics.org/releases/pixman-0.42.2.tar.gz && \
	tar -xf pixman-0.42.2.tar.gz && \
	cd pixman-0.42.2 && \
	mkdir build && \
	cd build && \
	meson --prefix=/usr --buildtype=release && \
	ninja && \
	ninja install && \
    rm -rf ../../pixman-0.42.2 && \
    rm -rf ../../pixman-0.42.2.tar.gz


# === Install python ===
# hadolint ignore=DL3003
RUN cd /tmp && \
    wget -q https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz && \
    tar xfv Python*.xz && \
    cd Python-3*/ && \
    ./configure --enable-shared LDFLAGS="-fprofile-arcs" && \
    make -j${BUILD_JOBS} install && \
    cd /tmp && \
    rm -rf Python-3* && \
    ldconfig

RUN pip3 install --no-cache-dir --upgrade pip==23.2.1 && \
    pip3 install --no-cache-dir cython packaging

# hadolint ignore=DL4006
RUN env | grep '^PATH=\|^LD_LIBRARY_PATH=\|^LANG=\|^LC_ALL=\|^CUDA_ROOT=' > /etc/environment

### seems to be needed, otherwise can't find some libcuda libs,
# but also seems to require in the python script to check for these libraries - very strange
ENV PATH="${PATH}:/usr/lib/x86_64-linux-gnu/"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
ENV LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH"

RUN ldconfig

# === Setup user ===
RUN useradd -ms /bin/bash user && \
    (echo user ; echo user) | passwd user && \
    gpasswd -a user sudo

USER user
WORKDIR /home/user

# === Install python libraries ===
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

#  === Install ASAP ===
# hadolint ignore=DL3003
RUN wget -q 'https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb' -P /tmp/ && \
    mkdir -p /home/user/libs && \
    dpkg -x /tmp/ASAP-2.2-Ubuntu2204.deb /tmp/asap && \
    mv /tmp/asap/opt/ASAP /home/user/libs/asap && \
    rm /tmp/ASAP-2.2-Ubuntu2204.deb && \
    rm -rf /tmp/asap && \
    ls /home/user/libs/asap
#     && echo "export PATH=\"/home/user/libs/asap/bin:\${PATH}\"" >> /home/user/.bashrc \
#     && echo "export PYTHONPATH=\"\${PYTHONPATH}:/home/user/libs/asap/bin\"" >> /home/user/.bashrc

ENV PATH="/home/user/libs/asap/bin:${PATH}"
ENV PYTHONPATH="${PYTHONPATH}:/home/user/libs/asap/bin"
RUN echo PATH:$PATH && echo PYTHONPATH:$PYTHONPATH

# === Set entrypoint  ===
COPY run.sh /home/user/run.sh
COPY run_pack_slides.py /home/user/run_pack_slides.py

# Compute requirements for the processor
LABEL processor.cpus="2"
LABEL processor.memory="15G"

ENTRYPOINT ["/bin/bash", "/home/user/run.sh"]
