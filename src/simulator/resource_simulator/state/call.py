class Call():
    """
    Describes a function call
    """

    def __init__(self, callee, *args, **kwargs):
        self._callee = callee  # Function to callback
        self._args = args
        self._kwargs = kwargs

    def call(self):
        if self._callee is None:
            return
        self._callee(*self._args, **self._kwargs)

    def to_string(self):
        parameters_str = ''
        if self._args is not None:
            i = len(self._args)
            for parameter in self._args:
                parameters_str += str(parameter)
                if i > 1:
                    parameters_str += ', '
                i -= 1

        return self._callee.__name__ + '(' + parameters_str + ')'
