# ensure Python 3.10.14 is available
# brew install python@3.10
# sudo apt install python3.10

# set up venv
pyenv local 3.10.14
python -m venv .calvin_utils_venv
source .calvin_utils_venv/bin/activate
pip install -r requirements.txt && pip install -e .