# Python 3.7

import logging
from libs.WikipediaScraper import WikiTableScraper
from libs.PoolScraper import PoolScraper
import unidecode
import pandas as pd
import os
import lxml
import lxml.html
from pathlib import Path
import sys
from difflib import SequenceMatcher
import numpy as np
from libs.CompetitionModifiers import BaseCompModifier

# Initializes logging file
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
#logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))


class BestFightOddsScraper(PoolScraper):
    def __init__(self, df, workers=4):
        super().__init__(workers=workers)
        self.urls = []
        self.df = df

    def run(self, upcoming=False):
        fighters = set(self.df['fighter'].values.tolist())
        fighters = [f.lower() for f in fighters]

        # Custom name changes
        # Geoff Neal and Geoffrey Neal are in bfo site
        # Ronaldo Souza is Jacare Souza in bfo site
        # Michelle Waterson-Gomez in ufcstats is Michaelle Waterson in bfo


        self.urls = [f'https://www.bestfightodds.com/search?query={f}' for f in fighters]
        search_responses = self.scrape(self.urls)
        fighter_page_urls = self.get_fighter_page_urls(search_responses)
        fighter_odds_responses = self.scrape(fighter_page_urls)
        odds_df = self.create_odds_df(fighter_odds_responses)

        # Upcoming fights
        if upcoming:
            df = odds_df[odds_df['date'] >= pd.to_datetime('today')]
            return df

        # All fights for training
        df = self.combine_dfs(self.df, odds_df)
        # needs to be \\mma when running mma.py
        df.to_csv(str(Path().resolve().parent) + '\\mma\\data\\odds.csv', index=False)
        #df.to_csv(str(Path().resolve().parent) + '\\data\\odds.csv', index=False)
        df.drop(columns=['opp_odds'], inplace=True)

        return df

    def get_upcoming_odds(self, odds_df):
        """Get upcoming fights from odds_df"""
        odds_df = odds_df[odds_df['date'] >= pd.to_datetime('today')]
        odds_df = odds_df.sort_values(by='date')


    def ufcstats_to_bfo_names(self, fighters):
        '''
        Change ufcstats names to bestfightodds names
        '''
        new_fighters = []
        for f in fighters:
            f = f.lower()
            if f == 'ovince saint preux':
                f = 'ovince st. preux'

    def combine_dfs(self, df, odds_df):
        df = df.reset_index(drop=True)

        # Normalize the original df
        df['date'] = pd.to_datetime(df['date'])

        # Keep only odds and opp odds
        odds_df = odds_df[['date', 'fighter', 'odds', 'opp_odds', 'opponent']]

        # Lowercase all fighters and opponents
        odds_df['fighter'] = odds_df['fighter'].str.lower()
        odds_df['opponent'] = odds_df['opponent'].str.lower()

        # Fix wrong bfo dates which are sometimes 1 day ahead of ufcstats
        dates = odds_df['date'].apply(lambda x: self.fix_date(x, df['date']))
        odds_df = odds_df.assign(date=dates)

        # Df is ordered by date. If date is the same, keep only latest row to filter out cancelled fights
        odds_df = self.remove_cancelled_fights(odds_df)

        # Check for fighter in consecutive rows
        # for row in df.itertuples():
        #     prev_i = row.Index - 1
        #     if row.fighter == df.iloc[prev_i]['fighter']:
        #         print(row)
        # Returns nothing of note! hooray!

        # Remove odds if it already exists in df
        if 'odds' in df.columns:
            df = df.drop(columns=['odds'])
        if 'opp_odds' in df.columns:
            df = df.drop(columns=['opp_odds'])

        df = df.merge(odds_df, how='left', on=['fighter', 'date', 'opponent'])

        # Check for odds in one row and not odds in opponent row
        df = self.fix_singular_odds(df)

        df = df.reset_index(drop=True)

        return df

    def fix_singular_odds(self, df):
        """Check and fix dangling odds, where one fighter in fight has odds and other doesn't
        How do I make this not use itertuples?"""
        df_copy = df.copy()
        for row in df_copy.itertuples():
            if pd.isna(row.odds):
                # frontfill
                if row.Index % 2 == 0:
                    if pd.notnull(df.iloc[row.Index + 1]['odds']):
                        df.loc[row.Index, 'odds'] = df.iloc[row.Index + 1]['opp_odds']
                        df.loc[row.Index, 'opp_odds'] = df.iloc[row.Index + 1]['odds']
                # backfill
                else:
                    if pd.notnull(df.iloc[row.Index - 1]['odds']):
                        df.loc[row.Index, 'odds'] = df.iloc[row.Index - 1]['opp_odds']
                        df.loc[row.Index, 'opp_odds'] = df.iloc[row.Index - 1]['odds']

        return df


    def remove_cancelled_fights(self, df):
        df.sort_values(by=['date', 'fighter'], inplace=True)
        df = df.drop_duplicates(subset=['date', 'fighter'], keep='first')
        return df

    def fix_date(self, x, df_date):
        td = pd.Timedelta(days=1)
        minus_one_day = x - td
        if minus_one_day in df_date.unique():
            return minus_one_day
        else:
            return x

    def get_fighter_page_urls(self, responses):
        fighter_page_urls = []
        for r in responses:
            html = r.text

            # Directly to fighter page
            if '/fighters/' in r.url:
                # Court mcgee shows up twice, why
                fighter_page_urls.append(r.url)
                continue

            # No match found
            elif 'No matching fighters or events found for search query' in html:
                fighter = r.url.split('=')[1].replace('+', ' ').replace('%20', ' ')
                #logging.info(f'No fighter found for {fighter}')
                continue

            # Search results
            else:
                url = self.parse_search_results(html)
                if url:
                    fighter_page_urls.append(url)

        return list(set(fighter_page_urls)) # this is for bug where 3 URLS show up identically like court mcgee

    def parse_search_results(self, html):
        # Sometimes search results show up for missing fighters
        url = None

        orig_fighter = html.split('Showing results for search query <b><i>')[1].split('</i></b>')[0]

        # fighter path in a href has dashes in place of spaces and dots
        path_fighter = self.get_fighter_path_name(orig_fighter)

        tree = lxml.html.fromstring(html)
        paths = tree.xpath('//table[@class="content-list"]/tr/td/a/@href')

        for path in paths:
            if path_fighter in path.lower():
                url = 'https://www.bestfightodds.com' + path
                return url

        #logging.info(f'No fighter found for {orig_fighter}')

        return url

    def get_fighter_path_name(self, fighter):
        # Fix for TJ/BJ/CJ/CB/AJ and remove Jr., O'Malley to O-Malley, Sangcha'An to Sangcha-An, T.J. O'Brien to T.J. Obrien, Ovince Saint Preux to Ovince St Preux, da'mon blackshear to da-mon
        replacements = [('tj ', 't-j '), ('bj ', 'b-j '), ('cj ', 'c-j '), ('cb ', 'c-b '), ('aj ', 'a-j '), ('jc ', 'j-c '),
                        ('jr.', ''), ("o'", 'o-'), ("a'a", "a-a"), ("o'brien", "obrien"), ('saint', 'st'), ("da'm", "da-m")]
        for r in replacements:
            fighter = fighter.replace(r[0], r[1])

        # href URLS use format Jim-Miller
        fighter = fighter.replace(' ', '-')

        return fighter

    def create_odds_df(self, responses):
        all_odds_df = pd.DataFrame()
        for r in responses:
            html = r.text

            # Sometimes see "Error 10"?
            if html.startswith('Error '):
                html = self.scrape([r.url])[0].text
                if html.startswith('Error '):
                    continue

            tree = lxml.etree.HTML(html)
            table = tree.xpath('//table[@class="team-stats-table"]')[0]
            table_html = lxml.etree.tostring(table)
            odds_df = pd.read_html(table_html)[0]
            odds_df = self.process_odds_df(odds_df)
            all_odds_df = pd.concat([all_odds_df, odds_df])

            # Drop dupes
            all_odds_df = all_odds_df.drop_duplicates(keep=False)

            all_odds_df.fillna(0, inplace=True)

        all_odds_df.reset_index(drop=True, inplace=True)

        return all_odds_df

    def process_odds_df(self, odds_df):
        # Drop the header row for each fight
        event_rows = odds_df[::3]
        dupe_df = pd.concat([odds_df, event_rows])
        odds_df = dupe_df.drop_duplicates(keep=False)

        # Remove fights with nan movement
        odds_df = odds_df[odds_df['Unnamed: 6'].isna() == False]
        odds_df = odds_df.reset_index(drop=True)

        # Jeff Monson never fought UFC
        if len(odds_df) == 0:
            return odds_df

        # Set date column
        odd_idx = odds_df[1::2]
        dates = odd_idx['Event'].values.tolist()
        dates = [val for val in dates for _ in (0, 1)] # double the date
        odds_df['date'] = dates

        # Remove special chars in Unnamed: 6 (movement)
        odds_df['Unnamed: 6'] = odds_df['Unnamed: 6'].apply(lambda x: self.process_movement(x))

        # Convert to numeric
        cols = ['Open', 'Closing range', 'Closing range.2', 'Unnamed: 6']
        for c in cols:
            odds_df[c] = pd.to_numeric(odds_df[c], errors='coerce')

        # Drop nonUFC
        odds_df_copy = odds_df.copy()
        for row in odds_df_copy.itertuples():
            if row.Index % 2 == 0:
                if 'UFC' not in row.Event:
                    odds_df.drop(row.Index, inplace=True)
                    odds_df.drop(row.Index + 1, inplace=True)
        odds_df = odds_df.reset_index(drop=True)

        # Convert odds to win%
        odds_df['Closing range'] = odds_df['Closing range'].apply(self.moneyline_to_win_perc)
        odds_df['Closing range.2'] = odds_df['Closing range.2'].apply(self.moneyline_to_win_perc)

        # Create avg closing range column
        odds_df['odds'] = odds_df[['Closing range', 'Closing range.2']].mean(axis=1)

        # # Create opponents column
        odd_idx = odds_df[1::2]
        opp_odds = odd_idx.odds.tolist()
        opp = odd_idx['Matchup'].values.tolist()
        odds_df.drop(index=odd_idx.index.tolist(), inplace=True)
        odds_df['opp_odds'] = opp_odds
        odds_df['opp'] = opp

        # Remove cancelled fights?
        #odd_idx = odds_df[1::2]
        #odds_df.drop(index=odd_idx.index.tolist(), inplace=True)

        # Remove "..." and Movement column which are irrelevant
        odds_df = odds_df.drop(columns=['Open', 'Closing range', 'Closing range.1', 'Closing range.2', 'Movement'])

        # Rename columns
        odds_df.columns = ['fighter', 'odds_movement_perc', 'event',  'date', 'odds', 'opp_odds', 'opponent']

        # Normalize data
        odds_df['date'] = pd.to_datetime(odds_df['date'])
        odds_df['fighter'] = odds_df['fighter'].astype(str)
        odds_df['opponent'] = odds_df['opponent'].astype(str)
        odds_df['fighter'] = odds_df['fighter'].str.lower()
        odds_df['opponent'] = odds_df['opponent'].str.lower()

        odds_df = odds_df.reset_index(drop=True)

        return odds_df

    def moneyline_to_win_perc(self, moneyline):
        if moneyline > 0:
            return 100 / (moneyline + 100)
        else:
            return moneyline / (moneyline - 100)

    def process_movement(self, movement):
        if movement == '0%':
            return 0
        else:
            return movement[:-2]


class BestFightOddsEventSearchStrings:
    def __init__(self):
        self.search_strings = []

    def process_events(self, df):
        df = df.iloc[::-1].reset_index(drop=True) # Reverse order to oldest first
        df = self.remove_old_fights(df) # Remove fights that bestfightodds doesn't have
        df = self.remove_cancelled(df) # remove cancelled fights
        df = self.normalize_event_name(df) # remove accented characters
        df = self.fix_ultimate_fighter_finals(df) # fix ultimate fighter finals
        df = df.fillna('0')
        return df

    def remove_old_fights(self, df):
        # Bestfightodds starts between UFC 73 and 72, at the Ultimate Fighter Finals Penn vs Pulver
        idx = df.index[df['Event'] == 'The Ultimate Fighter: Team Pulver vs. Team Penn Finale'].values[0]
        df = df[idx:].reset_index(drop=True)
        return df

    def remove_cancelled(self, df):
        df = df[df['#'] != '—']
        return df

    def normalize_event_name(self, df):
        res = df['Event'].apply(unidecode.unidecode)
        df['Event'] = res
        return df

    def fix_ultimate_fighter_finals(self, df):
        # BFO does "UFC: The Ultimate Fighter (#) Finale"
        df_copy = df.copy()
        count = 6
        for row in df_copy.itertuples():
            if "the ultimate fighter" in row.Event.lower():
                event = f"UFC: The Ultimate Fighter {str(count)} Finale"
                count += 1
                df.iloc[row.Index, df.columns.get_loc('Event')] = event
        return df

    def get_search_strings(self, df):
        search_strings = []

        events = df['Event'].to_list()
        #split_events = df['Event'].apply(lambda x: x.split())

        for event in events:
            split_event = event.split()
            print(event)

            # The Ultimate Fighter Finales
            if "the ultimate fighter" in event.lower():
                search_string = event

            # Major event, UFC 278
            elif split_event[1].replace(':', '').isnumeric():
                search_string = split_event[0] + ' ' + split_event[1].replace(':', '')

            # Minor event, UFC on ESPN: Vera vs. Cruz
            # We get vs. index, subtract one and take all the elements after that because of things like "UFC on ESPN: Vera vs. Cruz 2"
            else:
                if 'vs.' in split_event:
                    vs_id = split_event.index('vs.')
                    start = vs_id - 1
                    search_string = ' '.join(split_event[start:])

                # "UFC Fight for the troops" has no vs.
                else:
                    if 'fight for the troops' in event.lower():
                        if '3' in event:
                            search_string = 'UFC Fight Night 31: Fight For The Troops III'
                        elif '2' in event:
                            search_string = 'UFC Fight Night 23: Fight For The Troops II'
                        else:
                            search_string = 'UFC Fight Night 16: Fight For The Troops'

            search_strings.append(search_string)

        return search_strings

    def run(self, df):
        # Process events
        df = self.process_events(df)
        self.search_strings = self.get_search_strings(df)

        return self.search_strings






# if __name__ == '__main__':
    # wts = WikiTableScraper()
    # url = 'https://en.wikipedia.org/wiki/List_of_UFC_events'
    # tid = 'Past_events'
    # events_df = wts.run(url, tid)


    # url = 'https://en.wikipedia.org/wiki/List_of_UFC_events'
    # tid = 'Scheduled_events'
    # wts = WikiTableScraper(url, tid)
    # event_links = wts.get_table_links()

    # ps = PoolScraper()
    # urls = ['https://www.azlyrics.com/'] * 10
    # responses = ps.scrape(urls)

    # bfoss = BestFightOddsEventSearchStrings()
    # search_strings = bfoss.run(events_df)
    #
############################################################
    # all fights
    #df = pd.read_csv(/path/to/fights.csv')
    #bfos = BestFightOddsScraper(df)

    # Upcoming fights
    # df = pd.read_csv('/path/to/futurefights.csv')
    # bfos = BestFightOddsScraper(df)
    # resps = bfos.run(upcoming=True)









