import time

def do_task(name: str, delay: float) -> None:
    print(f"[START] Task {name}, will take {delay} seconds")
    time.sleep(delay)   # blocking
    print(f"[END]   Task {name}")

def main():
    start = time.perf_counter()

    # Run three tasks one after another
    do_task("A", 2)
    do_task("B", 2)
    do_task("C", 2)

    end = time.perf_counter()
    print(f"\nTotal time (sync): {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
