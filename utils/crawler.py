import time
from urllib.parse import urljoin, urlparse, urlunparse
import requests
from bs4 import BeautifulSoup

from playwright.sync_api import sync_playwright
import time


class FakeResponse:
    def __init__(self, url, content, status=200):
        self.url = url
        self._content = content.encode("utf-8") if isinstance(content, str) else content
        self.status_code = status

    @property
    def text(self):
        return self._content.decode("utf-8")

    @property
    def content(self):
        return self._content

def fetch_url(url, max_retries=2):
    # proxies = {
    #     "http": "127.0.0.1:7890",  # 替换为你的 HTTP 代理服务器
    #     "https": "127.0.0.1:7890",  # 替换为你的 HTTPS 代理服务器
    # }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    retries = 0  # 初始化重试次数
    while retries <= max_retries:
        try:
            # 发送请求并设置超时时间为n秒
            response = requests.get(
                url,
                headers=headers,
                verify=False,
                timeout=5,
                # proxies=proxies,
            )
            # 检查响应状态码
            if response.status_code == 200:
                print(f"{url} - OK (Status Code: {response.status_code})")
                return response
            elif response.status_code == 429:  # 访问频繁被限制
                print(f"{url} - Error (Status Code: {response.status_code})")
                retries += 1
                delay = 2 * retries + 60  # 退避
                print(
                    f"{url} - Rate limited (429). 访问频繁被限制，等待 {delay} 秒后重试 (Attempt {retries}/{max_retries})"
                )
                time.sleep(delay)
                print(f"Retrying... ({retries}/{max_retries})")
            else:
                print(f"{url} - Error (Status Code: {response.status_code})")
                break  # 非200状态码也退出循环
        except requests.exceptions.Timeout:
            print(f"Timeout occurred - {url}")
            retries += 1
            if retries <= max_retries:
                delay = 2 * retries + 5
                print(
                    f"{url} - Timeout, waiting {delay}s before retry ({retries}/{max_retries})"
                )
                time.sleep(delay)
                continue
            else:
                print(f"{url} - Max retries exceeded after timeout.")
                break
        except requests.exceptions.ConnectionError as e:
            print(f"ConnectionError: {url} - {e}")
            retries += 1
            delay = 2 * retries + 10  # 退避
            print(
                f"{url} - Connection failed...，等待 {delay} 秒后重试 (Attempt {retries}/{max_retries})"
            )
            time.sleep(delay)
            print(f"Retrying... ({retries}/{max_retries})")
        except requests.exceptions.MissingSchema as e:
            print(f"MissingSchema: {url} - {e}")
            break  # 对于 MissingSchema 异常，直接退出循环
        except Exception as e:
            print(f"Unexpected error: {url} - {e}")
            break  # 对于其他未捕获的异常，直接退出循环
    return None


def crawl_page(url):
    response = fetch_url(url=url, max_retries=3)
    if not response:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    # text = soup.get_text(separator="\n", strip=True)
    res = {"url": response.url, "text": text, "html": response.text}
    return res


def extract_navigation_links(base_url, nav_selectors=None):
    """
    从任意网页中提取导航区域（如 <nav> 或指定选择器）中的所有链接。

    参数:
        base_url (str): 要抓取的网页完整 URL。
        nav_selectors (list of str, optional):
            用于定位导航区域的 CSS 选择器列表。默认为 ["nav"]，
            即查找所有 <nav> 标签。可传入如 ["#navbar", ".sidebar-nav", "nav", ".docs-sidebar", "#main-navigation"] 等。
            如仅提取页眉导航栏：links = extract_navigation_links("https://example.com", nav_selectors=["header nav"])

    返回:
        list: 去重后的完整绝对链接列表（按出现顺序保留）。
    """
    if nav_selectors is None:
        nav_selectors = ["nav"]

    # 发起 HTTP 请求
    response = requests.get(base_url)
    response.raise_for_status()

    # 解析 HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # 使用集合保持唯一性，同时用列表保持顺序
    seen = set()
    links = []

    # 遍历每个选择器，查找导航区域
    for selector in nav_selectors:
        nav_elements = soup.select(selector)
        for nav in nav_elements:
            for a_tag in nav.find_all("a", href=True):
                href = a_tag["href"].strip()
                if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                    continue  # 忽略锚点、邮件、电话、JS 等非页面链接
                full_url = urljoin(base_url, href)
                if full_url not in seen:
                    seen.add(full_url)
                    links.append(full_url)

    return links


def normalize_url(url):
    # 解析URL
    parsed_url = urlparse(url)
    # 忽略查询参数和片段，重新构建URL
    # TODO 参数字符串，query应该要有的吧，但有时候不需要，这里可能引入一次
    # 问答？
    normalized_url = urlunparse(
        (parsed_url.scheme, parsed_url.netloc, parsed_url.path, "", "", "")
    )
    return normalized_url


def crawl_with_rec_filter(url, depth, domain=None, visited=None, valid=None):
    """递归访问（层数由depth指定）网页内所有链接，并过滤链接上的部分字段，以及只访问域名内的链接"""
    if visited is None:
        visited = set()
    if valid is None:
        valid = {}

    # filter
    # filter1: 去掉 参数字符串，查询字符串，片段  https://www.example.com/index.html;user?id=5#comment
    # ​scheme：URL 的协议部分（如 http 或 https）, ​netloc：网络位置部分（如 www.example.com）, ​path：URL 的路径部分（如 /index.html）, ​params：URL 的参数字符串（如 user）, ​query：URL 的查询字符串部分（如 id=5）, ​fragment：URL 的片段部分（如 #comment）
    url = normalize_url(url)
    # filter2: 域名之外的页面不访问
    if domain and urlparse(url).netloc != domain:
        return

    if url in visited:
        return
    print(f"Crawling: {url}")
    response = fetch_url(url=url, max_retries=2)
    # 注意使用response.url，因为原url如"https://celltypist.readthedocs.io/en/latest"实际上会访问"https://celltypist.readthedocs.io/en/latest/"链接，注意结尾多了个/
    visited.add(response.url if response else url)

    # TODO db存两种，一种有效的一种无效的url
    if not response:
        # invalid into db
        return
    soup = BeautifulSoup(response.text, "html.parser")
    # text = soup.get_text(separator="\n", strip=True)
    # text = soup.get_text(strip=True)
    # valid into db
    valid[response.url] = response.text

    if depth <= 1:
        return
    for link in soup.find_all("a", href=True):
        next_url = urljoin(response.url, link["href"])
        crawl_with_rec_filter(next_url, depth - 1, domain, visited, valid)


if __name__ == "__main__":
    r = fetch_url("https://scvi-tools.readthedocs.io/en/latest/")
    print(r.text)
