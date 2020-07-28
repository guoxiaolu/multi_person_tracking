import time
def run_time(func):  
    def wrapper(*args, **kw):  
        local_time = time.time()  
        value = func(*args, **kw) 
        print ('%s: %d ms' % (func.__module__ + '.' + func.__name__ ,(time.time() - local_time) * 1000))
        return value
    return wrapper 