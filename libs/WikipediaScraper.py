import requests
from bs4 import BeautifulSoup
from lxml import etree
import pandas as pd


class WikiTableScraper:
    def __init__(self, url, id):
        self.url = url
        self.id = id
        self.soup = self.get_soup(requests.get(self.url))
        self.base_url = self.get_base_url()
        self.table = self.get_table()

    # def run(self, url, tid):
    #     r = requests.get(url)
    #     soup = self.get_soup(r)
    #     events_df = self.get_table_by_id(soup, tid)
    #     return events_df


    def get_base_url(self):
        split = self.url.split('://', 1)
        proto = split[0] + '://'
        domain = split[1].split('/', 1)[0]
        return proto + domain

    def get_soup(self, r):
        soup = BeautifulSoup(r.text, "html.parser")
        return soup

    def get_table(self):
        table = self.soup.find('table', self.id)
        return table

    def get_table_by_id(self):
        pe_df = pd.read_html(str(self.table))[0]
        return pe_df

    def get_table_links(self, column):
        links = []
        for tag in self.table.select(f"td:nth-of-type({column}) a"):
            links.append(self.base_url + tag['href'])
        return links

    def get_table_column(self, column):
        data = []
        for tag in self.table.select(f"td:nth-of-type({column})"):
            data.append(tag.text.strip())
        return data

# url = 'https://en.wikipedia.org/wiki/List_of_UFC_events'
# id = {'id': 'Scheduled_events'}
# ws = WikiTableScraper(url, id)
# #events_df = ws.get_table_by_id()
# columns = '1'
# event_links = ws.get_table_links(column)






