#!env python3

import csv
from lxml import html
import requests

page = requests.get('https://docs.google.com/spreadsheets/d/e/2PACX-1vReSbN-lyLXPfYyyq3_wSZ7aRh8LHwbOtXl97jvfafnWpxqIgEiE5VxC6eEpm7Mt8WtV6ckqfn14i6N/pubhtml?gid=2145314735')

tree = html.fromstring(page.content)


with open('raid_secrets.tsv', 'w') as f:
    tsvwriter = csv.writer(f, delimiter='\t')
    for row in tree.xpath('//tr'):
        tsvwriter.writerow(row.xpath('td/text()'))
    
