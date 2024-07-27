# worker.py

import os
import asyncio
import aioredis
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")  # If your Redis requires authentication

# Construct Redis URL
REDIS_URL = f"redis://{':' + REDIS_PASSWORD + '@' if REDIS_PASSWORD else ''}{REDIS_HOST}:{REDIS_PORT}"

# Redis connection
redis: aioredis.Redis = None

async def connect_to_redis() -> None:
    global redis
    try:
        redis = await aioredis.from_url(REDIS_URL)
        await redis.ping()
        logger.info(f"Connected to Redis successfully at {REDIS_HOST}:{REDIS_PORT}")
    except aioredis.RedisError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise

async def process_job(job_id: str, job_data: Dict[str, Any]) -> None:
    # Update job status to PROCESSING
    await redis.hset(f"job:{job_id}", "status", "PROCESSING")
    
    # Here you would implement the actual transcription logic
    # For this example, we'll just simulate work with a delay
    logger.info(f"Processing job {job_id}: {job_data}")
    await asyncio.sleep(10)  # Simulate work
    
    # Update job status to COMPLETED
    await redis.hset(f"job:{job_id}", "status", "COMPLETED")
    logger.info(f"Completed job {job_id}")

async def worker_loop() -> None:
    while True:
        try:
            # Wait for a job in the queue
            _, job_id = await redis.brpop("job_queue")
            job_id = job_id.decode('utf-8')
            
            # Get job data
            job_data = await redis.hgetall(f"job:{job_id}")
            job_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in job_data.items()}
            
            # Process the job
            await process_job(job_id, job_data)
        except aioredis.RedisError as e:
            logger.error(f"Redis error: {e}")
            # Wait a bit before trying again
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error processing job: {e}")

async def main() -> None:
    await connect_to_redis()
    await worker_loop()

if __name__ == "__main__":
    asyncio.run(main())
