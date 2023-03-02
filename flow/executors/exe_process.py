# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import logging
import os
import signal
import subprocess


def exe_process(process_path, process_name, exe_args):
    if not os.path.exists(process_path):
        raise FileNotFoundError(process_path)

    p = subprocess.Popen(process_path + ' ' + exe_args, shell=True)

    def signal_handler(a, b):
        p.kill()

    signal.signal(signal.SIGINT, signal_handler)  # 2
    signal.signal(signal.SIGTERM, signal_handler)  # 15

    p.wait()
    if (p.poll() == 0):
        logging.info(process_name + ' Finish.')
        return True
    else:
        logging.error(process_name + ' Fail.')
        return False
