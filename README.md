# 2.5DRecon
2.5D reconstruction experiments

# Virtual environment setup
```
 git clone https://github.com/Mstrutov/2.5DRecon
 cd 2.5DRecon
```
Check python and pip versions:
- `python3 -V` 
- `pip -V` - older versions than 20.2.3 might raise errors during openCV installation

Install python3:
- `sudo apt-get update`
- `sudo apt-get install python3.6`

Install the python3-venv package:
- `sudo apt-get install python3-venv`

Install or update pip:
- `sudo apt-get install python3-pip`
- `python3 -m pip install --user --upgrade pip`

Set up the virtual environment:
- `python3 -m venv env`
- `source env/bin/activate`
- `pip install pip==20.2.3`
- `pip install -r requirements.txt`
