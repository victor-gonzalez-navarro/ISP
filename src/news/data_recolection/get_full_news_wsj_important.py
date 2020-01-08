import urllib.request
from pathlib import Path
import requests
import json
from bs4 import BeautifulSoup
import re
import time
from tqdm import tqdm


FILE_PATH = Path(__file__).resolve().parents[0]
IN_PATH = (FILE_PATH / '../../data/wsj_urls.json').resolve()


def print_lines(txt, line_length=90):
    while len(txt) > line_length:
        if txt[0] == ' ':
            txt = txt.strip()
        else:
            length = line_length + txt[line_length:].find(' ')
            print(txt[:length])
            txt = txt[length:]
    print(txt.strip())


def print_lines(txt, line_length=90):
    while len(txt) > line_length:
        if txt[0] == ' ':
            txt = txt.strip()
        else:
            length = line_length + txt[line_length:].find(' ')
            print(txt[:length])
            txt = txt[length:]
    print(txt.strip())


def get_html(url):
    cookies = {
        'DJSESSION': 'country%3Des%7C%7Ccontinent%3Deu%7C%7Cregion%3D%7C%7Ccity%3Dmadrid%7C%7Clatitude%3D40.40%7C%7Clongitude%3D-3.68%7C%7Ctimezone%3Dgmt%2B1',
        'wsjregion': 'europe%2Ces',
        'gdprApplies': 'true',
        'ccpaApplies': 'false',
        'ab_uuid': '430309c4-b532-494f-98ff-6abed1ab9f33',
        'usr_bkt': '2qgQ1WCa3A',
        'vidoraUserId': '2tdl53et9i62d9mk84n47u7ikn2ikl',
        'AMCVS_CB68E4BA55144CAA0A4C98A5%40AdobeOrg': '1',
        's_cc': 'true',
        '_ncg_sp_ses.5378': '*',
        '_ncg_g_id_': 'c2508cc4-d08e-41d7-a7f5-ed42a98b7c1b',
        '_scid': '582061ce-ec11-41f5-a802-f84f315ef689',
        '_fbp': 'fb.1.1576137375602.1016639736',
        '_mibhv': 'anon-1576137375697-4815604040_4171',
        'cX_P': 'k42fhyirlsk071ne',
        'cX_S': 'k42fhyj2hkynppw0',
        '_parsely_visitor': '{%22id%22:%22a14f9e6d-da4a-415b-9c6d-900cccd6b8b9%22%2C%22session_count%22:1%2C%22last_session_ts%22:1576137375994}',
        '_parsely_session': '{%22sid%22:1%2C%22surl%22:%22https://www.wsj.com/articles/iphone-se-review-smaller-gets-smarter-with-better-battery-life-to-boot-1458762252%22%2C%22sref%22:%22%22%2C%22sts%22:1576137375994%2C%22slts%22:0}',
        '__qca': 'P0-210318364-1576137375678',
        'OB-USER-TOKEN': 'da9447dd-7fd9-41d6-9bd7-0d79cfab976c',
        'cX_G': 'cx%3A1zywqhs0rpi1d2hbuy2gkohyid%3A31mvbtkete8b3',
        'hok_seg': 'none',
        'DJCOOKIE': 'EU_Cookie_Policy%3Dtrue%7C%7CtextSize%3Dregular',
        'djcs_route': '5cf391f7-2823-47e3-81a2-cb60fb837793',
        'TR': 'V2-09c88f3618bba8abc95eb2fb77e5f2bb312855a5876cefdf2e2780c385a3d450',
        'usr_prof_v2': 'eyJpYyI6NH0',
        'optimizelyEndUserId': 'oeu1576137772347r0.23075914526733898',
        'csrf_token': '278xIOi5-K8XOyYyxaGYc0ECjh7v-Uc4HPi4',
        's_vnum': '1607674081008%26vn%3D1',
        's_invisit': 'true',
        's_vmonthnum': '1577833200010%26vn%3D1',
        's_monthinvisit': 'true',
        'gpv_pn': 'DWSJN_Commerce_CAJ_Thank_You',
        'djcs_info': 'eyd1dWlkJzonYmM3NDk4NTAtOWVjZS00YzI1LTlhMzMtNTQzZTMxMWY3YjcyJywnZW1haWwnOidqc2llcnJhLnNpZXRlJTQwZ21haWwuY29tJywnZmlyc3RfbmFtZSc6J0pvcmdlJywndXNlcic6J2pzaWVycmEuc2lldGUlNDBnbWFpbC5jb20nLCcjdmVyJzonMScsJ2xhc3RfbmFtZSc6J1NpZXJyYScsJ3Nlc3Npb24nOicwNmFjOWVkNS0wOWVlLTRmYzYtOWYzYi0xMjQzNjEwNjgwOWMnLCdyb2xlcyc6J0ZSRUVSRUctSU5ESVZJRFVBTCd9',
        'djcs_auto': 'M1576098224%2FG6xabxp2Nq6%2BSCqmDQtNPMXWIEWNeOxpicytTt3cfLFlavfX2tFiqU%2Ba9pasD7vR6nKEHzmObZ6PUwajjvaipvuREyvwJ9CK1yzcNF9tN9puwaq5%2Fd21yok6wqca9uecG%2BjgmTgBfNT9CN7MKa44t27lqmrTDfEBcp%2Bmv9WBJ4%2B51QnxG9XxHvfHHVs1i%2F1s5DZzxalocjURS7Cxb1HyPjRjGyFFHaykjyJngn6a6syFIcakpWFAiKk8x%2Fm%2FvAmmp3S5jhsE38Atde1Y5dd7VPU5v8jw1hJ7l6fQhNeWhv5a%2B4qsrjbYYGCGntlopp%2FCz%2BGbEyHXXomqYQ9MrPEHIg%3D%3DG',
        'djcs_session': 'M1576133028%2F1KyR6tYXpnkZlbR3c2Ev2zSyp4TB3EgowjBoJqRqZnUANqFasEm7bwHz1nXbOSDY9sCNyphJJNvIqxFv%2BeXCLsLqsyVo8MJYAFLFnVucVszdv2R7MlKN9JRjKKoW8h2yiy3c8iTY1llOYAechiU%2B9RTcclH3G1DRZK6DX1gjBQEGnHdGxgERGGT%2B%2FVbxjPJjlVwpCrPfEdwNnFeJwjD%2BHTHaCVhTpV7xAM656jdrjj0onUPWuPb5cDQ%2FlKNiEzyoOtmca5cBKOIaBjrGXJxaXKAtbFyH3NvzySfWQROTvSgBI28GdSctlOm3GEXceFL2mgFSBWdllScY5%2FF%2Fjs12BWGEJc0j%2FOaV0lmuV8g%2BWph3Zt%2FHpuVOV9s%2F%2FnXwC7rmNr8lNy8tds4QIHPEV7jWq2aOmAb80stpUNhUd4U39LxDG9QYmLjgJp73Z1gwTq7glqcMaagWkJS5NOOVHXvSOElOOW%2FMIO34uHsZ4B1vup4%3DG',
        'djvideovol': '1',
        'AMCV_CB68E4BA55144CAA0A4C98A5%40AdobeOrg': '1585540135%7CMCIDTS%7C18243%7CMCMID%7C56016788136999233733518269725265017025%7CMCAAMLH-1576743001%7C6%7CMCAAMB-1576743001%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1576145401s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C4.4.0',
        'MicrosoftApplicationsTelemetryDeviceId': '8746eae8-b882-4e61-9fae-564460e66c41',
        'MicrosoftApplicationsTelemetryFirstLaunchTime': '1576138201361',
        '_micpn': 'esp:-1::1576137375697',
        '_ncg_id_': '16eed6f0600-773a139f-f265-4d2c-889e-a05ea4a5e865',
        '_ncg_sp_id.5378': '418fc9db-646e-4cb1-9e4a-ecb2ff56877a.1576137375.1.1576138356.1576137375.0fc8ee27-a7b5-4f1c-b119-fe5c9c3de9e4',
        '_tq_id.TV-63639009-1.1fc3': '3956905b7db81d88.1576137376.0.1576138356..',
        'utag_main': 'v_id:016ef91bbcc5007b402b88b1faa803084006e07c00bd0$_sn:1$_se:21$_ss:0$_st:1576140156698$ses_id:1576137374918%3Bexp-session$_pn:19%3Bexp-session$_prevpage:undefined%3Bexp-1576141956702$vapi_domain:wsj.com',
        's_sq': 'djglobal%3D%2526pid%253DWSJ_Article_Tech_Apple%252520to%252520Pay%252520%25252438%252520Billion%252520In%252520Repatriation%252520Tax%25253B%252520Plans%252520New%252520U.S.%252520Campus%252520%2526pidt%253D1%2526oid%253Dfunction%252528%252529%25257B%25257D%2526oidt%253D2%2526ot%253DDIV',
        's_tp': '11337',
        's_ppv': 'WSJ_Article_Tech_Apple%2520to%2520Pay%2520%252438%2520Billion%2520In%2520Repatriation%2520Tax%253B%2520Plans%2520New%2520U.S.%2520Campus%2520%2C9%2C9%2C1036'
        }

    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, cookies=cookies, headers=headers)
    return response.text


def main():

    with IN_PATH.open('r', encoding='utf-8') as f:
        data = json.load(f)
    data = [d for d in data if d['source'] != 'WSJ Video']

    new_data = []
    for d in tqdm(data):
        html = get_html(d['url'])

        parsed_html = BeautifulSoup(html, 'html.parser')

        title = parsed_html.find('h1', {'class': 'wsj-article-headline'})
        if title is None:
            title = parsed_html.find('h1', {'class': 'bigTop__hed'})
        if title is None:
            title = parsed_html.find('h1', {'class': 'article__headline'})

        desc = parsed_html.find('h2', {'class': 'sub-head'})
        if desc is None:
            desc = parsed_html.find('h2', {'class': 'bigTop__dek'})

        content = ''
        content_html = parsed_html.find('div', {'class': 'article-content'})
        if content_html is None:
            content_html = parsed_html.find('div', {'class': 'article__body'})

        paragraphs = content_html.find_all(recursive=False)
        start = False
        for p in paragraphs:
            if p.name == 'p' or p.name == 'h6':
                start = True
            if start and p.name != 'p' and p.name != 'h6':
                break
            if start:
                content += ' ' + p.text

        paragraphs = content_html.find('div', {'class': 'paywall'}).find_all('p')
        for p in paragraphs[:-1]:
            content += ' ' + p.text

        d['title'] = title.text.strip() if title else None
        d['desc'] = desc.text.strip() if desc else None
        d['content'] = re.sub(r'\s+', ' ', content).strip()
        new_data.append(d)

        time.sleep(1)

    with open('../../data/preprocessed_wsj.json', 'w') as f:
        json.dump(new_data, f, indent=4, sort_keys=True, default=str)


if __name__ == '__main__':
    main()
