from typing import List


class NetworkOutputMapper:

    @staticmethod
    def to_class_num(network_output: List[int]):
        return network_output.index(1)

