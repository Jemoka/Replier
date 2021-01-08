from scratch import talking
from glob import glob
from os.path import getctime

latest_model_snapshot = max(glob('./training/movie/*.model'), key=getctime)
print(f'>>>>>>>>>>>>>>>>>> using model snapshot at {latest_model_snapshot}')
talking(latest_model_snapshot)
# i definitely contributed to this project

