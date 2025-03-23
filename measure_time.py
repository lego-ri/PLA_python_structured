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
        print(f"\nSubroutine {func.__name__} starting...")
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

# import time
# import threading

# def measure_time(func):
#     """Decorator for measuring the time taken to execute a function."""
#     def measure_and_write_time(*args, **kwargs):
#         print(f"\nSubroutine {func.__name__} starting...")
#         start = time.time()  # Start measuring the total time
#         start_process = time.process_time()  # Start measuring the CPU time used
        
#         stop_event = threading.Event()  # Event to signal when to stop the status update thread
        
#         def status_update():
#             """Function to print elapsed time every 20 seconds until stopped."""
#             elapsed = 0
#             while not stop_event.is_set():  # Loop until stop_event is set
#                 time.sleep(20)  # Wait for 20 seconds
#                 elapsed += 20
#                 print(f"Function ongoing, elapsed time: {elapsed} s")
                
#         status_thread = threading.Thread(target=status_update, daemon=True)  # Create a background thread
#         status_thread.start()  # Start the thread
        
#         try:
#             result = func(*args, **kwargs)  # Execute the wrapped function
#         finally:
#             stop_event.set()  # Signal the thread to stop once the function completes
        
#         end = time.time()  # End total timer
#         end_process = time.process_time()  # End CPU time measurement
#         print(f"Subroutine {func.__name__} finished.")
#         print(f"Time taken to run {func.__name__} (total): {end - start:.5f} seconds")
#         print(f"Time taken to run {func.__name__} (processor time): {end_process - start_process:.5f} seconds")
        
#         return result  # Return the result of the wrapped function
    
#     return measure_and_write_time