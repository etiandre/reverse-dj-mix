import scrapy


class MixoticSpider(scrapy.Spider):
    name = "mixotic"
    allowed_domains = ["mixotic.net"]
    start_urls = ["https://www.mixotic.net/free-dj-sets/mixes.php"]

    def parse(self, response):
        for i in response.css(".mixlisthead"):
            yield response.follow(i, callback=self.parse_mixlist)

    def parse_mixlist(self, response):
        for i in response.css(".releaseLink"):
            yield response.follow(i, callback=self.parse_mix)

    def parse_mix(self, response):
        dl_link = response.xpath('//img[@src="images/downloadbutton.gif"]/..')[0]
        yield response.follow(dl_link, callback=self.parse_download)

    def parse_download(self, response):
        mp3, cover, txt = response.css(".downloadLink")
        id = int(response.url.split("/")[-1])
        return {
            "id": id,
            "file_urls": [
                response.urljoin(mp3.attrib["href"]),
                response.urljoin(cover.attrib["href"]),
                response.urljoin(txt.attrib["href"]),
            ]
        }
