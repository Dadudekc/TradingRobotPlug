import aiohttp
import asyncio
import pandas as pd

async def fetch_data(session, url):
    async with session.get(url) as response:
        return await response.json()

async def main():
    url = 'https://jsonplaceholder.typicode.com/posts'  # Example API URL
    async with aiohttp.ClientSession() as session:
        data = await fetch_data(session, url)
        
        # Assuming the data is a list of dictionaries
        df = pd.DataFrame(data)
        print(df)

if __name__ == "__main__":
    asyncio.run(main())
