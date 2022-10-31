import lxml.html
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import urllib
import lxml.html
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

dataframe = {'genre':[], 'label':[], 'date':[], 'artist':[], 'score': [], 'text':[]}


#Is passed an ablum review URL from get_reviews().  Scans url elements for key data points such as the album genre, label,
#date of reivew, artist, album score, and the review text.  This is done by finding the HTML classes associated with these
#elements in the html code across differnt album reviews.  Theses elements are appended to a dictionary.
def scrape(url):
    response = requests.get(url)

    response_text = response.text
    soup = BeautifulSoup(response_text, 'lxml')

    print(url)
    try:
        score = soup.find_all(['p', 'div'], {'class': rating(soup)})[0].text
    except:
        score = 'N/A'
    try:
        genre = soup.find_all('p', {'class' : 'BaseWrap-sc-UABmB BaseText-fETRLB InfoSliceValue-gTzwxg hkSZSE btxYx clvEtZ'})[0].text
    except:
        genre = 'N/A'
        print('missing gnre')
    try:
        label = soup.find_all('p', {'class' : 'BaseWrap-sc-UABmB BaseText-fETRLB InfoSliceValue-gTzwxg hkSZSE btxYx clvEtZ'})[1].text
    except:
        label = 'N/A'
        print('Missing label')
    try:
        date = soup.find_all('p', {'class' : 'BaseWrap-sc-UABmB BaseText-fETRLB InfoSliceValue-gTzwxg hkSZSE btxYx clvEtZ'})[2].text
    except:
        date = 'N/A'
        print('missing date')
    try:
        artist = soup.find_all('div', {'class' : 'BaseWrap-sc-UABmB BaseText-fETRLB SplitScreenContentHeaderArtist-lfDCdQ hkSZSE FFNqX PRayn'})[0].text
    except:
        artist = 'N/A'
        print('missing artist')
    try:
        text = soup.find_all ('div', {'class' : 'body__inner-container'})[0].text
    except:
        text = 'N/A'
        print('missing text')

    dataframe['genre'].append(genre)
    dataframe['label'].append(label)
    dataframe['date'].append(date)
    dataframe['artist'].append(artist)
    dataframe['score'].append(score)
    dataframe['text'].append(text)

def rating(soup):
    elements = soup.find_all(['p', 'div'])
    for element in elements:
        try:
            cl = ' '.join(element.attrs['class'])
            if 'Rating' in cl:
                return cl
        except:
            continue

#Is passed an artist page url from get_artists. Finds all url's within the html of the artist page that refer to album
#reviews.  Passes the specic album review URL to scrape.
def get_reviews(artist_url):
    artist_url = artist_url + 'albumreviews'
    artist_response = requests.get(artist_url)
    soup = BeautifulSoup(artist_response.text, 'lxml')
    links = soup.find_all('a')
    for link in links:
        href = link.get('href')
        if '/reviews/albums/' in href and href != '/reviews/albums/':
            full_link = 'https://pitchfork.com' + href
            scrape(full_link)

#Goes through links in the saved HTML that contain artist and appends them to the pitchfork url.  This new URL for the
#Specific arist is passed to get_reviews()
def get_artists(html):
    data = open(html, 'r', encoding="utf-8")
    soup = BeautifulSoup(data, 'html.parser')
    links = soup.find_all('a')
    for link in links:
        href = link.get('href')
        if '/artists/' in href and href != '/artists':
            new_link = 'https://pitchfork.com/' + href
            print(new_link)
            get_reviews(new_link)



#get_reviews('https://pitchfork.com/artists/3139-of-montreal/')
#dataframe = pd.DataFrame(dataframe)
#print(dataframe.head())


'''All Pitchfork reviews are contrained in different Genre pages.  From those pages you can access artists, and from those
artist pages you can access every review.  The artist list, however, exists on a infinitely scrolling javascript page and
to scrape these URLs selenium was used to scroll to the bottom of the page and then save the loaded HTML to a text document.
'''

chromedriver = 'C:\Datasets\chromedriver.exe'
os.environ['webdriver.chrome.driver'] = chromedriver
driver = webdriver.Chrome(chromedriver)
driver.get("https://pitchfork.com/artists/by/genre/jazz/")
ScrollNumber = 500
for i in range(1,ScrollNumber):
    driver.execute_script(f"window.scrollTo(1,{i*150})")
    time.sleep(0.3)

file = open('C:/PF/PFjazz.html', 'w', encoding="utf-8")
file.write(driver.page_source)
file.close()

driver.close()

get_artists('C:/PF/PFjazz.html')

df = pd.DataFrame(dataframe)
df.to_excel('C:/PF/PFjazz.xlsx')
















