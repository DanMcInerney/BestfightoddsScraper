import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor


class PoolScraper:
    def __init__(self, workers=3):
        self.failed = []
        self.responses = []
        self.workers = workers

    # Function to fetch data from server
    def fetch(self, session, base_url):
        with session.get(base_url) as response:
            #data = response.text
            if response.status_code != 200 or response.text.startswith('Error '):
                print("FAILURE::{0}".format(base_url))
                self.failed.append(base_url)
            else:
                if base_url in self.failed:
                    self.failed.remove(base_url)
            return response

    async def get_data_asynchronous(self, urls):
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            with requests.Session() as session:
                # Set any session parameters here before calling `fetch`
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(
                        executor,
                        self.fetch,
                        *(session, url) # Allows us to pass in multiple arguments to `fetch`
                    )
                    for url in urls
                ]
                self.responses = await asyncio.gather(*tasks)
                return self.responses

    def do_retries(self, retries, loop):
        while len(self.failed) > 0:
            retries = retries - 1
            self.retry(loop)
            if retries == 0:
                break

    def retry(self, loop):
        future = asyncio.ensure_future(self.get_data_asynchronous(self.failed))
        self.responses += loop.run_until_complete(future)


    def scrape(self, urls, retries=3):
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(self.get_data_asynchronous(urls))
        self.responses += loop.run_until_complete(future)

        # Retry failures
        self.do_retries(retries, loop)

        # Reset responses so we can reuse this object in more PoolScraper calls
        responses = self.responses
        self.responses = []

        return responses
