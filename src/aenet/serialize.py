"""
Serialization with JSON.

"""

import json

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"
__date__ = "2020-11-21"
__version__ = "0.1"


class Serializable(object):
    def to_JSON(self, indent=None):
        def serialize(o):
            if hasattr(o, '__dict__'):
                return o.__dict__
            else:
                try:
                    return o.tolist()
                except AttributeError:
                    return o
        return json.dumps(self, default=serialize, sort_keys=True,
                          indent=indent)

    @classmethod
    def from_JSON(cls, json_string):
        o = cls()
        o.__dict__.update(json.loads(json_string))
        return o
