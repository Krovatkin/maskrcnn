import subprocess

subprocess.run(["git submodule sync && git submodule update --init --recursive --jobs 0"], shell=True)
vision_dir = "./vision"
subprocess.check_call(["python", "setup.py", "install"], cwd=vision_dir) 
