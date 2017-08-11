#!/usr/bin/env python

import rospy
from rl_path_follow import PathFollow

def main():
    pf = PathFollow(load_model = False, record_data = False)

    pf.train_model(lr=0.00002, n_epochs=2000)

if __name__=='__main__':
    main()
