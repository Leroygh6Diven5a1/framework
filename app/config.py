from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    API_KEY: str = "123456"
    VERTEX_EXPRESS_API_KEY: Optional[str] = None
    FAKE_STREAMING: bool = False
    FAKE_STREAMING_INTERVAL: float = 1.0
    MODELS_CONFIG_URL: str = ""
    ROUNDROBIN: bool = False
    SAFETY_SCORE: bool = False
    PROXY_URL: Optional[str] = None
    SSL_CERT_FILE: Optional[str] = None

    # Cookie 鐩磋繛妯″紡閰嶇疆锛堟帹鑽?Render 绛変簯绔儴缃蹭娇鐢級
    GOOGLE_COOKIE: Optional[str] = None         # Google Cookie 瀛楃涓诧紙浠庢祻瑙堝櫒 DevTools 澶嶅埗锛?    GOOGLE_PROJECT_ID: Optional[str] = None     # Google Cloud 椤圭洰 ID锛堜粠 Console URL 涓幏鍙栵級
    EXPERIMENT_FLAGS: Optional[str] = "A/gnoZK5N/X6uWXbcVIxCDlJDN3ca/U9NrzddR8JfTo5D6TE4EzGrQf1GWrSGwLAYo+xZuK79EbXQsILnOEQbSAhnhZKgrI9CPJ6MrHdWBl0TY6u075dy0QSg2wdiXqjJ7KLjmkaK5ERwKzcmq67XLiOhaxjetPdWVt5R2pw96ha8/iPu/3kb8gB40hPt9bYRJbDgxJH24RLLN/5r8gnRJNeAntPLNea2Txoc1+DQ8GL9VK6hTNfndphSZzkEyjY1HTiDVBxGkCYewXMJa07Z7qIG86RhZcCVb7aWZj8HhrYwwDU5x8zySfkhlMP6otZnYYeUZXXmw6jr8iK7wdqZ8F4HN4h2Y2kWXHMRIA7oFzCZwZt+r4Kf5+u17pQgjXTjRsL48257kfVCIlopMTwkPLxZEV82DmjpP7Ev11laFiyIGFToy7P9GHNICURtP3MAaVeJnCNlpqs4XlK9waBB2qn2i1fne1+3/Vo960L4adLbdPFlBV/KlJN4oNg36JN5KBd59WtQGGYlzaKXFp+gAojzSxRwdm6kILMyH8sxn1pSJdtOYFfFvU8Gl0E/kRhZ4nnkTsBKh5hCTcczKr9/mF7ZJD3yfVE0xvMUwY/nK9xDJ6rcGLErABmWN3GVoSC5c8sxlvGRIW90JRs2kBgp1CnZu0SPNmrYn0sshgwLSUDUYPlQBuclvIdwREp8+RMD7fp2UFJTPY8Naqzx6XqJdY5mDlR8LFC64pQLIls27IAQzjdtbDH1hpI56vv/R5Vz3KfHKxNewKuCrwsqKquS5QXVI7MIkaLEmeCK/eJnaky90bp9UDrbdHKQo0LCGIYnN+vx8Cyv/1hZzyPvHUN77STmZYStfgJKzPv4VJN/ES1TEMsjIG0MwiNlj+moRpn5yAaI40+U62bWi52XbH9mcCEyGXGqLGotyndM5YwdSNyVzHTzXtu/RhXtbE+TvoIzvDjh/HTqe7bIj7ZtmALlzd2/eynH5xuwLithoKxPwyD+Cw009+QDPrZbQqF16c/8dmU/ycvZ6jfd7mTsKesx8FOBTfk2+gsH4XIuYwZQpFFYVgRYb+1FajvLvEVxqRwLjqoRJ5LlrQ3l4ZGd0UP4YBUN9MfDEPDgX2IFsjkF6OogOx34NC0eamckxXVMDgh3ZBa9VkUE9ymUlJBzxPjrjAuxHnLDnUWE1rYvLJk6MSJV82q4TXJgH0IUOlhTZfozV/M0k9cvV4+x72wa2Vf2wgcLHtpLFl4p2QRL8O2BFyXthZ6ed93YHS3LzJW861+zLYZ3xr3cuQkl2pNHm4kTG/AfP0/llFELug+nvfYcikdmvvdnZpKRaiH7p3miNrG/2tp6fjrXlJZ9DDnE99U8fiy2AFmtLUhpQ2aex2LoC/zsXOnBu+zrbriyLEsDGeUGA1/4lZz1bVI2bnNnIFnMWWIqUM8273Lt5fm/8ISDGM40N1nZBazmQ9eoeqyNYbKpaOBMEvDPGu40YIh2YJdhWi7wb1KXbM3SaHIRom5aahHEH5lSV5/6f8QFjJFrdb8cHaLu8YshFVNPakPP2Pzqe/uDx/xI3nQfThL9VIwLvLJComsXA5qs3X5CB/0anAbjAjSiSJKjBJQ8YsCKB6Rd
<truncated 1764 bytes>
aDFDp/hXUp16tqMrk+sw8EAMa3v77jwrZJi9pFI9vuE/vdfR9M9nQfeSxsw2zoAjGxrv/ohGORGmEeTPsDLfahhLa4rTTN7xP6CCHI8e9m3MY/+v9QBTrP557BarMmITse1cb2ny5F1yTmU4CxnnDVKxRT+o82j6hD0bqdFIzE7UdpfG7j5cvTm+oiL1F/UekdMmKSYNePS67AoHMDZz/HLfdVphSYuVhBVUIcRlUhrts084zyx6coThDYJg2IZ4/u+4BepQrCqqHRgiGa7c0h6kyzyzrl/7wnaKyiL+pfJXZTTSKu5UnyKeM0PsMpNE+DruZJefu86zsZa81gI9LKRwS92WSRl1e57Sm0PcelKQIQ5FeLrLUJ7MB1SV9f7vCxiMOXB2vlYAH0AK48a2R9hgYbbvUQRSgpHD5XN4o34TRHaxuQ1p8OJOaGzqnKck0kid4g1lnP8Jrpr249NYq4PYJnRO54y5UzrnsNtAh7R9rEA8WnezdATY33+DORfiTyLg+TRynU6bgxArHhW9vzv2pDRSZuA8aDeeEO6zMnIne6A8aCg715CzMmNnOyA8aDWPHHczMmNnO2A8qDlG4fJzMiNmJTD9ob3Yli/3LzN59THbyDFDumuqPvL59fHfDFEb+mvqPmxpdMNtclY2PqYlNT2iPPTKb/JiKrgkua55JmMqeyTj6iBjpOo/Z+YsIGU2/bw4FWWv9T7zJrs/syZ6vrL59fHuEEhl+mvmYuxptOc3rwS2P/vm56659XH5LqHgOmpnI+v7A=="       # experimentFlagsBinary锛堜粠 F12 Network 涓幏鍙栵紝Express Mode 鏉冮檺鏍囪瘑锛?
    # 鏃犲ご娴忚鍣ㄦā寮忛厤缃紙浠呮湰鍦伴儴缃插彲鐢級
    HEADLESS_MODE: bool = True
    CREDENTIAL_REFRESH_INTERVAL: int = 180

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


_settings = AppSettings()

API_KEY = _settings.API_KEY

raw_vertex_keys = _settings.VERTEX_EXPRESS_API_KEY
if raw_vertex_keys:
    VERTEX_EXPRESS_API_KEY_VAL = [key.strip() for key in raw_vertex_keys.split(",") if key.strip()]
else:
    VERTEX_EXPRESS_API_KEY_VAL = []

FAKE_STREAMING_ENABLED = _settings.FAKE_STREAMING
FAKE_STREAMING_INTERVAL_SECONDS = _settings.FAKE_STREAMING_INTERVAL
MODELS_CONFIG_URL = _settings.MODELS_CONFIG_URL
ROUNDROBIN = _settings.ROUNDROBIN
SAFETY_SCORE = _settings.SAFETY_SCORE
PROXY_URL = _settings.PROXY_URL
SSL_CERT_FILE = _settings.SSL_CERT_FILE

GOOGLE_COOKIE = _settings.GOOGLE_COOKIE
GOOGLE_PROJECT_ID = _settings.GOOGLE_PROJECT_ID
EXPERIMENT_FLAGS = _settings.EXPERIMENT_FLAGS
HEADLESS_MODE = _settings.HEADLESS_MODE
CREDENTIAL_REFRESH_INTERVAL = _settings.CREDENTIAL_REFRESH_INTERVAL

VERTEX_REASONING_TAG = "vertex_think_tag"


