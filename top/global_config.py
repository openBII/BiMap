# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

import logging

import toml


class GlobalConfig:
    Config = toml.load("top/config.toml")
    Path = Config['path']
    ChipArch = Config['chip_arch']
    Memory = Config['memory']
    Computation = Config['computation']
    Router = Config['router']

    @staticmethod
    def config():
        # 设置Python日志输出等级
        logging.basicConfig(level=logging.INFO)
