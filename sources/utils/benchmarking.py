import time


def measure_time(func: callable) -> callable:
    """
    Decorator function to measure the execution time of a given function.

    :param func: The function to measure the execution time of.
    :type func: callable
    :return: The wrapped function.
    :rtype: callable
    """

    def wrapper(*args, **kwargs):
        """
        Wrapper function that measures the execution time of the decorated function.

        :param args: Positional arguments to pass to the decorated function.
        :type args: tuple
        :param kwargs: Keyword arguments to pass to the decorated function.
        :type kwargs: dict
        :return: The result of the decorated function.
        """

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time:.2f} seconds")
        return result

    return wrapper
