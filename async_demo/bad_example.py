# file: asyncio_demo.py
import asyncio


async def bad_task():
    print("[bad] starting long CPU loop")
    x = 0
    # Long, CPU-heavy loop, but notice: NO await inside
    for _ in range(50_000_000):
        x += 1
    print("[bad] finished CPU loop")


async def polite_task():
    for i in range(5):
        print(f"[polite] step {i}")
        await asyncio.sleep(0.5)  # cooperative yield point
    print("[polite] done")


async def main():
    asyncio.create_task(bad_task())
    asyncio.create_task(polite_task())


asyncio.run(main())


# if __name__ == "__main__":
#    asyncio.run(main())
