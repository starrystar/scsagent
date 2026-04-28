# === 第一阶段：基础系统 + Miniconda 全局安装（root）===
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 替换 apt 源为阿里云
RUN sed -i 's|http://[a-z0-9\.]*archive\.ubuntu\.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security\.ubuntu\.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list

# 安装必要系统包
RUN apt update && \
    apt install -y --no-install-recommends ca-certificates sudo vim git wget build-essential python3-dev curl iputils-ping && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# === 安装 Miniconda 到 /opt（全局只读）===
COPY Miniconda3-py310_25.5.1-0-Linux-x86_64.sh /tmp/
RUN bash /tmp/Miniconda3-py310_25.5.1-0-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm /tmp/Miniconda3-py310_25.5.1-0-Linux-x86_64.sh

# 设置 PATH（所有用户可用 conda 命令）
ENV PATH="/opt/miniconda3/bin:${PATH}"

# === 全局 pip 配置（使用 CERNET 源）===
RUN echo "[global]" > /etc/pip.conf && \
    echo "index-url = https://mirrors.cernet.edu.cn/pypi/web/simple" >> /etc/pip.conf && \
    echo "trusted-host = mirrors.cernet.edu.cn" >> /etc/pip.conf

# === conda 配置：使用 .condarc 文件完全配置镜像源 ===
RUN echo "auto_activate_base: false" > /opt/miniconda3/.condarc && \
    echo "channel_priority: strict" >> /opt/miniconda3/.condarc && \
    echo "show_channel_urls: true" >> /opt/miniconda3/.condarc && \
    echo "channels:" >> /opt/miniconda3/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge" >> /opt/miniconda3/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda" >> /opt/miniconda3/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main" >> /opt/miniconda3/.condarc && \
    echo "default_channels:" >> /opt/miniconda3/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main" >> /opt/miniconda3/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r" >> /opt/miniconda3/.condarc

# === 第二阶段：创建普通用户 ===
RUN groupadd -g 1013 STOmics_test && \
    useradd -u 1013 -g STOmics_test -m -s /bin/bash STOmics_test && \
    echo "STOmics_test ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER STOmics_test
WORKDIR /home/STOmics_test

RUN conda init bash
# 创建用户私有环境
RUN conda create -p /home/STOmics_test/.conda/envs/py310 -y python=3.10 && \
    conda run -p /home/STOmics_test/.conda/envs/py310 pip install --no-cache-dir \
        Cython scanpy leidenalg umap-learn

# 自动激活
RUN echo "conda activate /home/STOmics_test/.conda/envs/py310" >> ~/.bashrc
RUN git config --global http.version HTTP/1.1

ENV PATH="/home/STOmics_test/.conda/envs/py310/bin:/opt/miniconda3/bin:${PATH}"
CMD ["/bin/bash", "-l"]
