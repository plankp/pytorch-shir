# PyTorch Stuff

## Build and Setup

Note that since the project uses `torch.compile`, it is subject to the same platform restrictions

```sh
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

You would need to `source` every time.
Use `deactivate` to leave the venv for this project.

For development purposes, it might also be beneficial to run `pip install --editable .` once after setting up the environment and installing the dependencies.

