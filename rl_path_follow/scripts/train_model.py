#!/usr/bin/env python

import rospy
from rl_path_follow import PathFollow

def main():
    pf = PathFollow(load_model = False, record_data = False)

    pf.train_model(lr = 1e-3, n_epochs=800)

if __name__=='__main__':
    main()
