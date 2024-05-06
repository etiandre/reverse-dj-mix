# Complete mixotic dataset

This project scrapes the mixotic.net netlabel, downloads all mixes and associated metadata, and formats it in a machine-readable form.

## Recrating the dataset

```shell
# using a virtualenv is recommended
pip install -r requirements.txt
# download everything. check crawl.log for any errors and restart the crawl if needed
scrapy crawl mixotic -t jsonlines -o mixotic-crawl.jsonl 2> crawl.log
# cleanup results
python clean_mixotic.py > clean.log 2>&1
```

This outputs `mixotic.json`