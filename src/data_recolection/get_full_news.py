from src.sentiment_analysis.ground_truth import read_news, read_stocks
from pathlib import Path
from datetime import datetime
import urllib.request
import re


FILE_PATH = Path(__file__).resolve().parents[0]
IN_PATH = (FILE_PATH / '../../data/preprocessed.json').resolve()

# SITES
SITES = [
"6thfloor.blogs.nytimes",
"artsbeat.blogs.nytimes",
"atwar.blogs.nytimes.co",
"bats.blogs.nytimes.com",
"bits.blogs.nytimes.com",
"boss.blogs.nytimes.com",
"bucks.blogs.nytimes.co",
"carpetbagger.blogs.nyt",
"cityroom.blogs.nytimes",
"dealbook.blogs.nytimes",
"dealbook.nytimes.com/2",
"dinersjournal.blogs.ny",
"dotearth.blogs.nytimes",
"economix.blogs.nytimes",
"empirezone.blogs.nytim",
"essay.blogs.nytimes.co",
"executivesuite.blogs.n",
"firstlook.blogs.nytime",
"freakonomics.blogs.nyt",
"frugaltraveler.blogs.n",
"gadgetwise.blogs.nytim",
"goal.blogs.nytimes.com",
"green.blogs.nytimes.co",
"greeninc.blogs.nytimes",
"ideas.blogs.nytimes.co",
"india.blogs.nytimes.co",
"intransit.blogs.nytime",
"kristof.blogs.nytimes.",
"krugman.blogs.nytimes.",
"laughlines.blogs.nytim",
"learning.blogs.nytimes",
"mediadecoder.blogs.nyt",
"news.blogs.nytimes.com",
"op-talk.blogs.nytimes.",
"parenting.blogs.nytime",
"pogue.blogs.nytimes.co",
"rendezvous.blogs.nytim",
"runway.blogs.nytimes.c",
"schott.blogs.nytimes.c",
"sinosphere.blogs.nytim",
"sports.blogs.nytimes.c",
"takingnote.blogs.nytim",
"thecaucus.blogs.nytime",
"thelede.blogs.nytimes.",
"themoment.blogs.nytime",
"tmagazine.blogs.nytime",
"topics.blogs.nytimes.c",
"walkthrough.blogs.nyti",
"well.blogs.nytimes.com",
"wheels.blogs.nytimes.c",
"wordplay.blogs.nytimes",
"www.nytimes.com/2005/1",
]


def html_regex_constructor(tags_to_remove):
    regex = '<(?:{:s}).*?>(.*?)<\/.*?>'.format('|'.join(tags_to_remove))
    return re.compile(regex, re.IGNORECASE | re.DOTALL)


REGEX_TAGS_REMOVE = html_regex_constructor(tags_to_remove=['a', 'abbr', 'b', 'blockquote', 'cite', 'dfn', 'em', 'i', 'ins', 'li', 'mark', 'ol', 'q', 's', 'span', 'strong', 'sub', 'u'])
REGEX_TAGS_REMOVE_ALL = html_regex_constructor(tags_to_remove=['img', 'canvas', 'code', 'table', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'meta', 'style'])
REGEX_TAGS_COMMENTS = re.compile(r'<!--.*?-->', re.DOTALL)
REGEX_TAGS_BR = re.compile(r'<\/br>', re.DOTALL)
REGEX_TAGS_HR = re.compile(r'<\/hr>', re.DOTALL)
REGEX_TAGS_IMG = re.compile(r'<img.*?\/>', re.DOTALL)
REGEX_TAGS_SCRIPT = re.compile(r'<script.*?>.*?</script>', re.DOTALL)
REGEX_SYMBOLS = re.compile(r'&#\d+;', re.IGNORECASE)
REGEX_CONTENT = re.compile(r'^(([^>]*?</?p.*?>)+)')
REGEX_PARAGRAPH_INDICATOR = re.compile(r'</?p.*?>', re.IGNORECASE | re.DOTALL)
REGEX_MULTISPACE = re.compile(r'\s+')
REGEX_TEXT_NYT = re.compile(r'"text":"(.+?)","formats', re.DOTALL)
REGEX_BRACKETS = re.compile(r'\[.*?\]', re.DOTALL)


def remove_divs(html):
    modified = False
    paragraph_inside = False
    start = None
    index = 0

    while index < len(html):
        if html[index:].startswith('<div'):
            start = index
            index += 4
        if html[index:].startswith('</div>') and start is not None:
            end = index + 6
            html = html[:start] + html[end:]
            index -= end - start
            modified = True
            start = None

        index += 1

    return html, modified


def regex_spaces(txt):
    txt = re.sub(r'\s+', ' ', txt)
    txt = txt.replace(' ', '\\s+')
    return txt


def get_full_text(article, comparison_size=25):
    # Download HTML
    html = None
    req = urllib.request.Request(article['url'])
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8')

    # NYT format -----------------------------------
    if '"text":"' in html and '"formats":[' in html:
        content = ' '.join(REGEX_TEXT_NYT.findall(html))

    # HTML format -----------------------------------
    else:

        # Find lead paragraph position
        if not article['lead_paragraph']:
            raise Exception('No lead paragraph found')

        # Remove useless tags
        html = REGEX_TAGS_SCRIPT.sub('', html)
        html = REGEX_SYMBOLS.sub('', html)
        html = REGEX_TAGS_COMMENTS.sub(' ', html)
        html = REGEX_TAGS_BR.sub(' ', html)
        html = REGEX_TAGS_HR.sub(' ', html)
        html = REGEX_TAGS_IMG.sub(' ', html)
        html = REGEX_TAGS_REMOVE_ALL.sub(' ', html)
        html = REGEX_TAGS_REMOVE.sub(' \\1 ', html)

        result = re.findall('<p.*?>(.*?)</p>', html, re.DOTALL)
        cleaned = []
        for r in result:
            r = REGEX_BRACKETS.sub('', r)
            r = REGEX_TAGS_IMG.sub('', r)
            r = REGEX_MULTISPACE.sub(' ', r)
            cleaned.append(r)
        return cleaned

        # start = re.search(r'>\s*' + regex_spaces(article['lead_paragraph'][:comparison_size]), html).start() + 1
        # # start = html.find('>' + article['lead_paragraph'][:comparison_size]) + 1
        # cropped = html[start:]
        #
        # # Remove divs starting from the origin of the text
        # while True:
        #     cropped, modified = remove_divs(cropped)
        #     if not modified:
        #         break
        #
        # # Get chunk of text
        # result = REGEX_CONTENT.search(cropped)
        #
        # if result:
        #     content = result.group(1)
        # else:
        #     raise Exception('No content found')
        #
        # content = REGEX_PARAGRAPH_INDICATOR.sub('', content)

    # Final fixes
    content = REGEX_BRACKETS.sub('', content)
    content = REGEX_TAGS_IMG.sub('', content)
    content = REGEX_MULTISPACE.sub(' ', content)
    return [content]


def print_lines(txt, line_length=90):
    while len(txt) > line_length:
        if txt[0] == ' ':
            txt = txt.strip()
        else:
            length = line_length + txt[line_length:].find(' ')
            print(txt[:length])
            txt = txt[length:]
    print(txt.strip())


def main():
    news = read_news(IN_PATH)

    min_date = datetime(year=2005, month=12, day=30)
    # news = [article for article in news if article['date'] > min_date]

    from time import sleep
    from tqdm import tqdm
    import json
    updated = []
    errors = []
    i = 0
    for k, article in enumerate(tqdm(list(reversed(news)))):
        try:
            article['p'] = get_full_text(article)
        except:
            errors.append(article['url'])
            article['p'] = None
        sleep(2)
        updated.append(article)

        if k > 0 and (k % 200) == 0:
            with open('../../data/preprocessed' + str(i) + '.json', 'w') as f:
                json.dump(news, f, indent=4, sort_keys=True, default=str)
                i+=1

    with open('../../data/preprocessed' + str(i) + '.json', 'w') as f:
        json.dump(news, f, indent=4, sort_keys=True, default=str)
        i+=1

    for error in errors:
        print(error)

    # for article in news:
    #     if article['url'].startswith('https://' + SITES[1]):
    #         print(article['url'])

    # article = news[4]  # 100, 1, 0, 2, 3
    # print(article['url'])
    # content = get_full_text(article)
    # print_lines(content)
    # print(article['url'])
    # print(len(content))
    #
    # lengths = {100: 496, 0: 4950, 1: 1508, 2: 983, 3: 1143}


if __name__ == '__main__':
    main()
