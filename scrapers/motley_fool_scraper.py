import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from time import sleep
import argparse
# #%%
# r = requests.get('https://www.fool.com/investing-news/?page=1')
#
# #%%
# soup = BeautifulSoup(r.content, 'html5lib')
# articles_div = soup.find_all('div', attrs={'id': 'article_listing'})
# articles = articles_div[0].find_all('a')


#%%
class MotleyFoolScraper:

    LOG_LEVEL = 2

    URL_PREFIX = 'https://www.fool.com'
    THREAD_SLEEP_TIME = lambda self: np.random.rand()+0.5
    PAGE_SLEEP_TIME = lambda self: np.random.randint(1, 15)

    def __init__(self, storage_path):
        self.storage_path = storage_path
        # self.storage = pd.read_csv(storage_path, index_col=None)
        # self._save_storage()

    def _save_storage(self):
        self.storage.to_csv(self.storage_path, index=False)

    def _append_storage(self, new_data):
        pd.DataFrame(new_data).to_csv(self.storage_path, mode='a', index=False, header=False)

    def start_scrape(self, start_page=1, num_pages=0):
        page_idx = start_page
        new_data = list()
        while num_pages == 0 or page_idx <= num_pages:

            if self.LOG_LEVEL >= 1:
                print("Starting page", page_idx)

            failed_attempts_left = 10

            for page_url in self._fetch_article_list(page=page_idx):
                if not page_url.startswith('/investing'):
                    continue

                if self.LOG_LEVEL == 2:
                    print("Scraping", page_url)

                data = self._scrape_article(page_url)

                if data is not None:
                    new_data.append(data)
                    # self.storage.loc[len(self.storage)] = data
                else:
                    if self.LOG_LEVEL >= 1:
                        print("Failed on article link", self._complete_url(page_url))
                    if failed_attempts_left == 0:
                        print("Fatally failed on page", page_idx)
                        return
                    else:
                        failed_attempts_left -= 1

                sleep(self.THREAD_SLEEP_TIME())

            # self._save_storage()
            self._append_storage(new_data)
            new_data.clear()
            sleep(self.PAGE_SLEEP_TIME())
            page_idx += 1

    def _complete_url(self, url):
        if not url.startswith(self.URL_PREFIX):
            return self.URL_PREFIX + url
        else:
            return url

    def _fetch_soup(self, url):
        url = self._complete_url(url)

        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html5lib')
        return soup

    def _fetch_article_list(self, page=1):
        r = requests.get(f'https://www.fool.com/investing-news/?page={page}')
        soup = BeautifulSoup(r.content, 'html5lib')
        articles = soup.find('div', attrs={'id': 'article_listing'}).find_all('a')
        return [article['href'] for article in articles]

    def _scrape_article(self, url):
        soup = self._fetch_soup(url)

        title = soup.find_all('h1')
        if len(title) != 1:
            if self.LOG_LEVEL >= 1: print("Got irregular size of 'h1' title tag list:", len(title))
            return None
        title_text = title[0].text

        date = soup.find_all('div', attrs={'class': 'publication-date'})
        if len(date) != 1 and len(date) != 2:
            if self.LOG_LEVEL >= 1: print("Got irregular size of 'publication-date' class list:", len(date))
            return None
        date_text = date[0].text.strip()
        if len(date) == 2:
            date_text += '---'+date[1].text.strip()

        body = soup.find_all('div', attrs={'class': 'main-col'})

        if len(body) != 1:
            if self.LOG_LEVEL >= 1: print("Got irregular size of 'main-col' class list:", len(body))
            return None

        body_text_list = list()

        for p in body[0].find_all('p'):
            if len(p.text.strip()) > 0:
                body_text_list.append(p.text.strip())
        body_text = '\n---\n'.join(body_text_list)

        companies = soup.find_all('div', attrs={'class': 'ticker-row'})
        companies_list = list()
        for company in companies:
            company_list_entry = [
                company.find('h4').text.strip(),  # Ticker
                company.find('h3').text.strip()  # Name
            ]
            companies_list.append(company_list_entry)

        return {
            'title': title_text,
            'url': url,
            'date': date_text,
            'body': body_text,
            'companies': companies_list
        }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--page', type=int, default=0, help='Page to start scraping')
    args = parser.parse_args()

    scraper = MotleyFoolScraper('../Datasets/motleyfool/scraped.csv')
    scraper.start_scrape(start_page=args.page)
