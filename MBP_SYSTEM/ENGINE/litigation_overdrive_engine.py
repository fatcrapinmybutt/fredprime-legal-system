import os
import time
import concurrent.futures

THREAD_COUNT = os.cpu_count() * 10
TEMP_PATH = 'MBP_SYSTEM/VFS/TEMP'


def task(i):
    with open(os.path.join(TEMP_PATH, f"thread_{i}.log"), 'w') as f:
        f.write(f"Task {i} OK")
    time.sleep(0.1)
    return f"Done {i}"


def run():
    print(f'Threadstorm using {THREAD_COUNT} workers')
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as exe:
        list(exe.map(task, range(100)))
    print('Completed Thread Tasks')

if __name__ == '__main__':
    run()
