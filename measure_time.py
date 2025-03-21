""" 
    Decorator used to measure the time a function takes to be executed.
"""

#* Import necessary libraries 
import time

def measure_time(func):
    """ 
        Decorator for measuring the time taken to execute a function
    """
    def measure_and_write_time(*args, **kwargs):     # args and kwargs to enable handling of any number of arguments
        print(f"\n Subroutine {func.__name__} starting...")
        start = time.time()                          # Start measuring the total time
        start_process = time.process_time()          # Start measuring the time used by the processor
        result = func(*args, **kwargs)               # The original wrapped function is called and the results stored
        end = time.time()                            # End the total timer
        end_process = time.process_time()            # End measuring the time used by the processor
        print(f"Subroutine {func.__name__} finished.")
        print(f"Time taken to run {func.__name__} (total): {end - start:.5f} seconds")
        print(f"Time taken to run {func.__name__} (processor time): {end_process - start_process:.5f} seconds")
        return result                                # Return the result of the actual function which we want to measure time for
    return measure_and_write_time   