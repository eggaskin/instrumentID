import json
import os
import time

import requests 
import time 
from multiprocessing import cpu_count 
from multiprocessing.pool import ThreadPool

HOME_DIRECTORY = "C:/Users/evely/Desktop/cs270/instrumentID/datasets/" # path to instrumentID folder

def url_to_filepath(url, output_dir):
    output_file = os.path.join(output_dir,url.split('https://')[-1].split('/')[5],url.split('https://')[-1].split('/')[6],url.split('https://')[-1].split('/')[-1])
    outdir = os.path.dirname(output_file)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return output_file


def download_url(url, filename, skip_existing=True):
    t0 = time.time() 
    if os.path.exists(filename) and skip_existing:
        print(" Skipping (exists): {}".format(url))
        return True

    try: 
        r = requests.get(url) 
        with open(filename, 'wb') as f: 
            f.write(r.content) 
            return(url, time.time() - t0) 
    except Exception as e: 
        print('Exception in download_url():', e)

def download_many(urls, output_files, skip_existing=True):
    if len(urls) != len(output_files):
        raise ValueError(
            "Number of URLs ({}) does not match the number of output files "
            "({})".format(len(urls), len(output_files)))
        
    pairs = zip(urls, output_files)
    cpus = cpu_count() 
    results = ThreadPool(cpus - 1).imap_unordered(download_url, pairs) 
    for result in results: 
        print('url:', result[0], 'time (s):', result[1])

urls = json.load(open("uiowa.json"))['resources']
output_files = [url_to_filepath(url, HOME_DIRECTORY+"uiowa/") for url in urls]
success = download_many(urls, output_files,
                            skip_existing=True,
                            num_cpus=-1)


