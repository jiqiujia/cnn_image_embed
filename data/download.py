# -*- coding: utf-8 -*-
import io
import os
import urllib.request
import shutil

with io.open('pics.txt', 'r', encoding='utf-8') as fin:
    cnt = 0
    for line in fin.readlines():
        print(cnt)
        print(line)
        arr = line.strip().split('\t')
        if len(arr) != 3:
            continue
        cat = arr[1].split(',')[0].replace("/", "")
        url = arr[2]
        filename = url.split("/")[-1] + ".jpg"

        if len(cat) == 0:
            continue
        if not os.path.exists(cat):
            os.mkdir(cat)

        filepath = os.path.join(cat, filename)
        if os.path.exists(filepath):
            cnt += 1
            continue
        try:
            with urllib.request.urlopen(url, timeout=10) as response, io.open(filepath, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except Exception as e:
            print(url)
            print(e)
        cnt += 1