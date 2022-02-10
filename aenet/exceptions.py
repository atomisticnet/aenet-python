class ArgumentError(Exception):

    def __init__(self, msg):
        self.msg = msg


class IncompatibleStructureError(Exception):

    def __init__(self, msg="Structures are incompatible."):
        self.msg = msg


class InternalError(Exception):

    def __init__(self, msg):
        self.msg = msg


class FormatError(Exception):

    def __init__(self, frmt):
        self.msg = "Format not supported: {}".format(frmt)


class FormatGuessError(Exception):

    def __init__(self, filename):
        self.msg = "Failed to guess format of file: {}".format(filename)


class ReadonlyError(Exception):

    def __init__(self, frmt):
        self.msg = ("No write support implemented for file "
                    "format `{}'.".format(frmt))


class WriteonlyError(Exception):

    def __init__(self, frmt):
        self.msg = ("No read support implemented for file "
                    "format `{}'.".format(frmt))
