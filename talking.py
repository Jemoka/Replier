from scratch import talking
from glob import glob
from os.path import getctime

latest_model_snapshot = max(glob('./training/movie/*.model'), key=getctime)
if __name__ == '__main__':
    print(f'>>>>>>>>>>>>>>>>>> using model snapshot at {latest_model_snapshot}')
    talking(latest_model_snapshot)
# i definitely contributing to this project

