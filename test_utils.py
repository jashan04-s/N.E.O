
import time

def logTime(label, start_time):
    print(f"{label} took {time.time() - start_time:.2f} seconds")