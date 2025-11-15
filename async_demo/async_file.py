import asyncio
import time


async def do_task(name: str, delay: float) -> None:
    print(f"[START] Task {name}, will take {delay} seconds")
    await asyncio.sleep(delay)   # non-blocking wait
    print(f"[END]   Task {name}")


async def main():
    start = time.perf_counter()

    # Create 3 tasks that run concurrently
    tasks = [
        asyncio.create_task(do_task("A", 2)),
        asyncio.create_task(do_task("B", 2)),
        asyncio.create_task(do_task("C", 2)),
    ]

    # Wait for all of them to finish
    await asyncio.gather(*tasks)

    end = time.perf_counter()
    print(f"\nTotal time (async): {end - start:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())


# in multi-threading we've multiple threads running at the same time, # each thread can handle blocking operations independently.
# in async programming, we have a single thread that switches between tasks during blocking operations.