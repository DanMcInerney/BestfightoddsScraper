import requests
from bs4 import BeautifulSoup
from lxml import etree
import pandas as pd


class WikiTableScraper:

    def run(self, url, tid):
        r = requests.get(url)
        soup = self.get_soup(r)
        events_df = self.get_table_by_id(soup, tid)
        return events_df

    def get_soup(self, r):
        soup = BeautifulSoup(r.text, "html.parser")
        return soup

    def get_table_by_id(self, soup, tid):
        past_events = soup.find('table', {'id': tid})
        pe_df = pd.read_html(str(past_events))[0]
        return pe_df


ws = WikiTableScraper()
url = 'https://en.wikipedia.org/wiki/List_of_UFC_events'
tid = 'Past_events'
events_df = ws.run(url, tid)