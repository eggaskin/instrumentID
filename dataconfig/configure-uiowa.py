import json
import os
import time

import requests 
import time 
from multiprocessing import cpu_count 
from multiprocessing.pool import ThreadPool

HOME_DIRECTORY = "C:/Users/evely/Desktop/cs270/instrumentID/datasets/" # path to instrumentID folder
instruments = {'sopranosaxophone':'soprano-sax', 'piano':0, 'violin':0, 'doublebass':'bass','tenortrombone':'trombone','cello':0,'cymbals':0,'Xylophone':'xylophone','viola':0,'tuba':0,'guitar':0,'Bbclarinet':'clarinet','Bbtrumpet':'trumpet','flute':0,'altosaxophone':'alto-sax','basstrombone':'trombone','Ebclarinet':'clarinet','altoflute':'flute','oboe':0,'bassclarinet':'clarinet','bassoon':0}

def urls_to_filepaths(urls, output_dir):
    output_files = []
    urlss = []

    for url in urls:
        filefull = url.split('https://')[-1]
        instrument = filefull.split('/')[6]
        if instrument in instruments:
            instrument = instruments[instrument] if instruments[instrument] != 0 else instrument
        else:
            next
        filename = filefull.split('/')[-1]
        output_file = os.path.join(output_dir,instrument,filename)
        outdir = os.path.dirname(output_file)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        output_files.append(output_file)
        urlss.append(url)

    return output_files, urlss


def download_url(args, skip_existing=True):
    url, filename = args
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

urls = json.load(open("uiowatest.json"))['resources']
output_files,urls = urls_to_filepaths(urls, HOME_DIRECTORY+"uiowa/")
# remove None values
success = download_many(urls, output_files)

