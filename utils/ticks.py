# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
from typing import List

def full_universe(cfg) -> List[str]:
    uni = []
    for k in ('equities','bonds','commodities','crypto'):
        uni.extend(cfg['universe'].get(k, []))
    return uni