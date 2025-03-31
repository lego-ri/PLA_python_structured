import time
import threading

def measure_time(func):
    """Decorator for measuring the time taken to execute a function."""
    def measure_and_write_time(*args, **kwargs):
        print(f"\nFunction {func.__name__} starting...")
        start = time.time()  # Start measuring the total time
        start_process = time.process_time()  # Start measuring the CPU time used
        
        stop_event = threading.Event()  # Event to signal when to stop the status update thread
        
        def status_update():
            """Function to print elapsed time every 20 seconds until stopped."""
            # print(f"Status update thread started for {func.__name__}")
            elapsed = 0
            while not stop_event.is_set():  # Loop until stop_event is set
                time.sleep(20)  # Wait for 20 seconds
                elapsed += 20
                print(f"Function {func.__name__} ongoing, elapsed runtime: {elapsed} s")
                
        status_thread = threading.Thread(target=status_update, daemon=True)  # Create a background thread
        status_thread.start()  # Start the thread
        
        try:
            result = func(*args, **kwargs)  # Execute the wrapped function
        finally:
            stop_event.set()
            # print(f"Stop event set for {func.__name__}")
            status_thread.join()  # Wait for the thread to terminate
            # print(f"Status update thread stopped for {func.__name__}")
        
        end = time.time()  # End total timer
        end_process = time.process_time()  # End CPU time measurement
        print(f"Function {func.__name__} finished")
        print(f"Time taken to run {func.__name__} (total): {end - start:.5f} seconds")
        print(f"Time taken to run {func.__name__} (processor time): {end_process - start_process:.5f} seconds")
        
        return result  # Return the result of the wrapped function
    
    return measure_and_write_time