import json

instruments = {'sopranosaxophone':'soprano-sax', 'piano':0, 'violin':0, 'doublebass':'bass','tenortrombone':'trombone','cello':0,'cymbals':0,'Xylophone':'xylophone','viola':0,'tuba':0,'guitar':0,'Bbclarinet':'clarinet','Bbtrumpet':'trumpet','flute':0,'altosaxophone':'alto-sax','basstrombone':'trombone','Ebclarinet':'clarinet','altoflute':'flute','oboe':0,'bassclarinet':'clarinet','bassoon':0}

HOME_DIRECTORY = "C:/Users/evely/Desktop/cs270/instrumentID/datasets/" # path to instrumentID folder

def urls_to_filepaths(urls, output_dir):
    output_files = []
    urlss = []
    counts = {}

    for url in urls:
        filefull = url.split('https://')[-1]
        instrument = filefull.split('/')[6]
        if instrument in instruments:
            instrument = instruments[instrument] if instruments[instrument] != 0 else instrument
            if instrument not in counts:
                counts[instrument] = 0
            counts[instrument] += 1
        else:
            next

    return counts

urls = json.load(open("uiowa.json"))['resources']
counts = urls_to_filepaths(urls, HOME_DIRECTORY+"uiowa/")

print(counts)