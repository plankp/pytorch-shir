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

Note that the requirements point to a particular nightly PyTorch build,
which is probably going to be always outdated.
In that case, just go to the official PyTorch website and grab the latest nightly.
