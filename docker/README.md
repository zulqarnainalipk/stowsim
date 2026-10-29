# `docker/` – runtime environment

| File | Contents |
| --- | --- |
| `Dockerfile` | Ubuntu 24.04 image with Miniconda (Python 3.12), build tools, `gymnasium`, `pybullet`, `pillow` and CPU-only PyTorch |
| `docker-compose.yml` | Service `stowsim`: builds the image with the repository as build context and mounts the repository at `/workspace` |

```bash
docker compose -f docker/docker-compose.yml up -d
docker exec -it stowsim_env bash
cd /workspace && python -m scripts.run_test
```

Extra libraries for an agent go into the agent's own `requirements.txt`; install them inside the
container with `pip install -r agents/<name>/requirements.txt`.
