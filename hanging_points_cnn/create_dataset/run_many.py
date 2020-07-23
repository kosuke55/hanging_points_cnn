#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import signal
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', type=str)
    parser.add_argument('--jobs', '-j', type=int, default=10,
                        help='number of parallel process')
    parser.add_argument('--num-exp', type=int, default=10,
                        help='number of trial')
    parser.add_argument('--sleep-time', type=int, default=2)
    args = parser.parse_args()

    start = time.time()
    try:
        procs = []
        num_finish = 0
        for i in range(args.num_exp):
            cmd = args.cmd
            procs.append(subprocess.Popen(cmd, shell=True))
            print('start')
            time.sleep(args.sleep_time)
            while len(procs) >= args.jobs:
                for p in procs:
                    ret = p.poll()
                    if ret is not None:
                        procs.remove(p)
                        num_finish += 1
                        print("finish {} processes".format(num_finish))
                        break
                else:
                    time.sleep(args.sleep_time)
        while procs:
            for p in procs:
                ret = p.poll()
                if ret is not None:
                    procs.remove(p)
                    num_finish += 1
                    print("finish {}processes".format(num_finish))
                    break
            else:
                time.sleep(args.sleep_time)
    except KeyboardInterrupt:
        for p in procs:
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
    elapsed_time = time.time() - start
    print("elapsed_time: {0}".format(elapsed_time) + "[sec]")


if __name__ == '__main__':
    main()
