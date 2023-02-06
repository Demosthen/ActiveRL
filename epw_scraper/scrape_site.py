import requests
from pprint import pprint
import os
import urllib.request 

URL = "https://energyplus.net/weather"

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from time import sleep
opts = Options()
opts.add_argument('--headless')
opts.headless = True
assert opts.headless  # Operating in headless mode
browser = Chrome(options=opts)


def process_main_page(browser):
    browser.get(URL)
    sleep(1)
    elements = browser.execute_script("return document.getElementsByTagName(\"a\")")
    
    links = []
    for elem in elements:
        link = elem.get_attribute('href')
        if "weather-region" in link:
            links.append(link)
    return links

def process_subpage(browser, subpage_link):
    browser.get(subpage_link)
    sleep(0.25)
    elements = browser.execute_script("return document.getElementsByTagName(\"a\")")
    links = []
    for elem in elements:
        link = elem.get_attribute('href')
        if subpage_link + "/" in link:
            links.append(link)
    return links

def process_subsubpage(browser, subsubpage_link):
    browser.get(subsubpage_link)
    sleep(0.25)
    elements = browser.execute_script("return document.getElementsByTagName(\"a\")")
    links = []
    has_subregions = False
    for elem in elements:
        link = elem.get_attribute('href')
        if "weather-location" in link:
            links.append(link)
        if subsubpage_link + "/" in link:
            links.append(link)
            has_subregions = True
    return links, has_subregions

def process_download_page(browser, download_link):
    print(download_link)
    browser.get(download_link)
    sleep(0.25)
    elements = browser.execute_script("return document.getElementsByTagName(\"a\")")
    
    links = []
    for elem in elements:
        link = elem.get_attribute('href')
        if "epw" in link:
            filename = os.path.join("data", link.split("/")[-1])
            print(filename)
            if os.path.exists(filename):
                continue
            with open (filename, "wb") as f:
                f.write(requests.get(link).content)
            links.append(link)
    return links

links = process_main_page(browser)
for link in links[::-1]:
    sublinks = process_subpage(browser, link)
    for sublink in sublinks[::-1]:
        subsublinks, has_subregions = process_subsubpage(browser, sublink)
        if has_subregions:
            for subsublink in subsublinks[::-1]:
                subsubsublinks, _ = process_subsubpage(browser, subsublink)
                for subsubsublink in subsubsublinks:
                    process_download_page(browser, subsubsublink)
        else:
            for subsublink in subsublinks[::-1]:
                process_download_page(browser, subsublink)