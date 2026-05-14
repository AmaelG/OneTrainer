# To build, run:
# docker build \
#   --build-arg ONETRAINER_REPO=https://github.com/YOUR_USERNAME/OneTrainer.git \
#   --build-arg ONETRAINER_BRANCH=your-branch-name \
#   -t <image-name> . -f Vast-NVIDIA-CLI.Dockerfile

FROM vastai/pytorch:cuda-12.8.1-auto

WORKDIR /
USER root

ARG ONETRAINER_REPO=https://github.com/AmaelG/OneTrainer.git
ARG ONETRAINER_BRANCH=anima

RUN git clone \
      --branch ${ONETRAINER_BRANCH} \
      --single-branch \
      ${ONETRAINER_REPO} \
      /OneTrainer

RUN cd /OneTrainer \
 && export OT_PLATFORM_REQUIREMENTS=requirements-cuda.txt \
 && export OT_LAZY_UPDATES=true \
 && export OT_PYTHON_CMD=/venv/main/bin/python \
 && ./install.sh \
 && pip cache purge \
 && rm -r ~/.cache/pip

RUN apt-get update --yes \
 && apt-get install --yes --no-install-recommends \
      ca-certificates \
      curl \
      gnupg \
 && mkdir -p /usr/share/keyrings \
 && curl -fsSL https://debian.griffo.io/EA0F721D231FDD3A0A17B9AC7808B4DD62C41256.asc \
    | gpg --dearmor --yes -o /usr/share/keyrings/debian-griffo-archive-keyring.gpg \
 && codename="$([ -r /etc/os-release ] && . /etc/os-release && echo "${VERSION_CODENAME:-${UBUNTU_CODENAME:-}}")" \
 && test -n "$codename" \
 && echo "deb [signed-by=/usr/share/keyrings/debian-griffo-archive-keyring.gpg] https://debian.griffo.io/apt ${codename} main" \
    > /etc/apt/sources.list.d/debian-griffo.list \
 && apt-get update --yes \
 && apt-get install --yes --no-install-recommends \
      joe \
      less \
      gh \
			rsync \
      yazi \
      zoxide \
      iputils-ping \
      nano \
      nethogs \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install nvitop \
 && pip cache purge \
 && rm -rf ~/.cache/pip

RUN mkdir -p /workspace \
 && mkdir -p /OneTrainer/workspace-cache/run \
 && ln -snf /OneTrainer /workspace/OneTrainer

RUN cat <<'EOF' >> /root/.bashrc
eval "$(zoxide init bash)"

function y() {
  local tmp="$(mktemp -t "yazi-cwd.XXXXXX")" cwd
  yazi "$@" --cwd-file="$tmp"
  if cwd="$(command cat -- "$tmp")" && [ -n "$cwd" ] && [ "$cwd" != "$PWD" ]; then
    z -- "$cwd"
  fi
  rm -f -- "$tmp"
}
EOF
