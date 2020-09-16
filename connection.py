import os
import re
import time
import json
import random
from urllib.parse import urljoin
import requests
from selenium import webdriver
from selenium.webdriver.common.proxy import Proxy, ProxyType


'''
proxy = Proxy({
    'proxyType': ProxyType.MANUAL,
    'httpProxy': ':',
    'sslProxy': ':'
})
'''
proxy = None


def relation_parse(data, domain):
    relations = list()
    for d in data:
        d = re.sub(re.compile('(\w+)(\n)(\w+)'), r'\1 \3', d).replace('\n', '')
        relation = list()
        relation.extend(re.findall(re.compile('[\w\.-]+@[\w\.-]*' + domain), d))
        for composed_relation in re.findall(re.compile('\{[\w\.\s,-]+\}@[\w\.-]*' + domain), d):
            relations.extend(list(
                map(lambda x: f'{x}@{domain}', re.findall(re.compile('[\w\.-]+'), composed_relation[:
                                                                                 composed_relation.index(
                                                                                     '@')]))))

        relations.append(relation)
    return [relation for relation in relations if type(relation) == list]


def scrap_relations(domain, depth):
    if os.environ['GECKO_DRIVER']:
        driver = webdriver.Firefox(executable_path=os.environ['GECKO_DRIVER'], proxy=proxy)
    else:
        driver = webdriver.Firefox(proxy=proxy)

    driver.set_window_size(1920, 1080)
    driver.get(f'https://scholar.google.com/scholar?start=0&q=%22%40{domain}%22+email&hl=en&as_sdt=0,5')
    relations = list()
    for i in range(depth):
        time.sleep(random.random() * 120)
        data = [d.text for d in driver.find_elements_by_class_name('gs_rs')]
        relations.extend(relation_parse(data, domain))
        try:
            driver.find_element_by_class_name('gs_ico_nav_next').click()
        except:
            print('depth stopped', i)
            break
    driver.close()
    return relations


def check_mail(mail):
    if os.environ['GECKO_DRIVER']:
        driver = webdriver.Firefox(executable_path=os.environ['GECKO_DRIVER'], proxy=proxy)
    else:
        driver = webdriver.Firefox(proxy=proxy)

    driver.set_window_size(1920, 1080)

    driver.get(urljoin('https://haveibeenpwned.com/unifiedsearch/', mail))
    time.sleep(random.random() * 120)
    try:
        driver.find_element_by_id('rawdata-tab').click()
        data = json.loads(driver.find_elements_by_class_name('data')[0].text)
    except:
        data = list()
    driver.close()
    return len(data)
